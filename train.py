import argparse
import numpy as np
import pandas as pd
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

from models.model import (
    SpliceDetectorModelA, SpliceDetectorModelB, SpliceDetectorModelC, 
    SpliceDetectorEnsemble, FocalLoss, 
    create_SpliceDetector_models, create_SpliceDetector_ensemble
)
from data.encode_data import encode_and_split_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def result_report(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        class_true = np.argmax(y_true, axis=1)
    else:
        class_true = y_true
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        class_pred = np.argmax(y_pred, axis=1)
        prob_pred = y_pred
    else:
        class_pred = y_pred
        prob_pred = y_pred
    
    class_report = classification_report(class_true, class_pred, 
                                       labels=[0, 1], 
                                       target_names=['site', 'non_site'], 
                                       output_dict=True)
    
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        auc = roc_auc_score(y_true, prob_pred)
    else:
        y_true_onehot = np.eye(2)[class_true]
        auc = roc_auc_score(y_true_onehot, prob_pred)
    
    acc = class_report['accuracy']
    err = 1 - acc
    pre = class_report['site']['precision']
    sn = class_report['site']['recall']
    sp = class_report['non_site']['recall']
    f1 = class_report['site']['f1-score']
    
    report = "Acc: {:.4f}, Pre: {:.4f}, Sn: {:.4f}, Sp: {:.4f}, Err: {:.4f}, F1: {:.4f}, Auc: {:.4f}".format(
        acc, pre, sn, sp, err, f1, auc)
    return report

class AdvancedLRScheduler:
    """高级学习率调度器"""
    def __init__(self, optimizer, scheduler_type='reduce_on_plateau', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-6),
                verbose=True
            )
        else:
            self.scheduler = None
    
    def step(self, epoch=None, val_loss=None):
        if self.scheduler is not None:
            if self.scheduler_type == 'reduce_on_plateau' and val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            return current_lr
        return self.initial_lr

def create_data_loader(x, y, batch_size, shuffle=True):
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    if len(x.shape) == 4:
        x = x.squeeze(-2)  
    
    print(f"Data loader input shape: {x.shape}")
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader

def train_single_model(model, model_name, train_loader, val_loader, criterion, 
                      optimizer, scheduler, epochs, device, verbose, 
                      save_model_path, early_stop_patience=15):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.argmax(dim=1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            target_labels = target.argmax(dim=1)
            correct += pred.eq(target_labels).sum().item()
            total += target.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.argmax(dim=1))
                
                val_total_loss += loss.item()
                pred = output.argmax(dim=1)
                target_labels = target.argmax(dim=1)
                val_correct += pred.eq(target_labels).sum().item()
                val_total += target.size(0)
                all_predictions.append(torch.softmax(output, dim=1).cpu())
                all_targets.append(target.cpu())
        
        val_loss = val_total_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        current_lr = scheduler.step(epoch, val_loss)
        
        if verbose:
            print(f'  Epoch {epoch+1}/{epochs}:')
            print(f'    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'    Learning Rate: {current_lr:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.scheduler.state_dict() if scheduler.scheduler else None,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, save_model_path)
            
            if verbose:
                print(f'    *** New best model saved: {val_acc:.2f}% ***')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping triggered for {model_name} after {epoch+1} epochs")
                break
    
    checkpoint = torch.load(save_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            final_predictions.append(torch.softmax(output, dim=1).cpu())
            final_targets.append(target.cpu())
    
    final_predictions = torch.cat(final_predictions)
    final_targets = torch.cat(final_targets)
    
    final_report = result_report(final_targets.numpy(), final_predictions.numpy())
    print(f"  {model_name} Final Performance: {final_report}")
    
    return model, best_val_acc, final_report

def plot_training_curves(histories, model_names, save_path):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green']
    
    for i, (history, name, color) in enumerate(zip(histories, model_names, colors)):
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], f'{color[0]}-', 
                label=f'{name} Train', linewidth=2, alpha=0.8)
        ax1.plot(epochs, history['val_losses'], f'{color[0]}--', 
                label=f'{name} Val', linewidth=2, alpha=0.8)
        ax2.plot(epochs, history['train_accs'], f'{color[0]}-', 
                label=f'{name} Train', linewidth=2, alpha=0.8)
        ax2.plot(epochs, history['val_accs'], f'{color[0]}--', 
                label=f'{name} Val', linewidth=2, alpha=0.8)
    
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def train_ensemble_models(args, train_dataset_path, val_dataset_path):
    train_dataset = np.load(train_dataset_path)
    x_train = train_dataset['x_train']
    y_train = train_dataset['y_train']

    val_dataset = np.load(val_dataset_path)
    x_val = val_dataset['x_val']
    y_val = val_dataset['y_val']

    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")

    train_loader = create_data_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = create_data_loader(x_val, y_val, args.batch_size, shuffle=False)

    epochs = args.num_train_epochs
    verbose = args.verbose

    model_a, model_b, model_c = create_SpliceDetector_models(num_classes=2)
    models = [model_a, model_b, model_c]
    model_names = ['SpliceDetector-A (ECA)', 'SpliceDetector-B (ECA+Trans)', 'SpliceDetector-C (CBAM)']
    
    for model in models:
        model.to(device)
    
    seeds = [42, 123, 456]
    learning_rates = [args.learning_rate, args.learning_rate * 0.8, args.learning_rate * 1.2]
    
    trained_models = []
    model_performances = []
    training_histories = []
    
    for i, (model, name, seed, lr) in enumerate(zip(models, model_names, seeds, learning_rates)):
        print(f"\nSetting random seed to {seed} for {name}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name} parameters: {total_params:,}")
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        if args.use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        scheduler = AdvancedLRScheduler(
            optimizer, 
            scheduler_type='reduce_on_plateau',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
        
        save_model_path = f"models/trained_models/{args.organism_name}_{args.splice_site_type}_{chr(ord('a')+i)}.pth"
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        
        trained_model, best_acc, report = train_single_model(
            model, name, train_loader, val_loader, criterion,
            optimizer, scheduler, epochs, device, verbose,
            save_model_path, early_stop_patience=15
        )
        
        trained_models.append(trained_model)
        model_performances.append(best_acc)
        checkpoint = torch.load(save_model_path, map_location=device)
        training_histories.append({
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'train_accs': checkpoint.get('train_accs', []),
            'val_accs': checkpoint.get('val_accs', [])
        })
    
    plot_path = f"results/SpliceDetectort_training_comparison_{args.organism_name}_{args.splice_site_type}.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_training_curves(training_histories, model_names, plot_path)
    total_performance = sum(model_performances)
    ensemble_weights = [perf / total_performance for perf in model_performances]
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE SUMMARY")
    print(f"{'='*60}")
    for name, perf, weight in zip(model_names, model_performances, ensemble_weights):
        print(f"{name:25} | Acc: {perf:.2f}% | Weight: {weight:.3f}")
    
    ensemble_model = create_SpliceDetector_ensemble(num_classes=2, ensemble_weights=ensemble_weights)
    ensemble_model.to(device)
    ensemble_model.model_a.load_state_dict(trained_models[0].state_dict())
    ensemble_model.model_b.load_state_dict(trained_models[1].state_dict())
    ensemble_model.model_c.load_state_dict(trained_models[2].state_dict())
    ensemble_model.eval()
    ensemble_predictions = []
    ensemble_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = ensemble_model(data)
            ensemble_predictions.append(torch.softmax(output, dim=1).cpu())
            ensemble_targets.append(target.cpu())
    
    ensemble_predictions = torch.cat(ensemble_predictions)
    ensemble_targets = torch.cat(ensemble_targets)
    ensemble_report = result_report(ensemble_targets.numpy(), ensemble_predictions.numpy())
    
    print(f"Ensemble Performance: {ensemble_report}")
    
    ensemble_save_path = f"models/trained_models/{args.organism_name}_{args.splice_site_type}_ensemble.pth"
    torch.save({
        'ensemble_weights': ensemble_weights,
        'model_a_state_dict': trained_models[0].state_dict(),
        'model_b_state_dict': trained_models[1].state_dict(),
        'model_c_state_dict': trained_models[2].state_dict(),
        'ensemble_performance': ensemble_report,
        'individual_performances': model_performances
    }, ensemble_save_path)
    
    return ensemble_report

def predict_ensemble(args, test_dataset_path, trained_model_path):
    test_dataset = np.load(test_dataset_path)
    x_test = test_dataset['x_test']
    y_test = test_dataset['y_test']

    print(f"Test data shape: {x_test.shape}")
    test_loader = create_data_loader(x_test, y_test, args.batch_size, shuffle=False)
    checkpoint = torch.load(trained_model_path, map_location=device)
    ensemble_weights = checkpoint['ensemble_weights']
    ensemble_model = create_SpliceDetector_ensemble(num_classes=2, ensemble_weights=ensemble_weights)
    ensemble_model.model_a.load_state_dict(checkpoint['model_a_state_dict'])
    ensemble_model.model_b.load_state_dict(checkpoint['model_b_state_dict'])
    ensemble_model.model_c.load_state_dict(checkpoint['model_c_state_dict'])
    ensemble_model.to(device)
    ensemble_model.eval()
    all_predictions = []
    all_targets = []
    individual_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            ensemble_output = ensemble_model(data)
            all_predictions.append(torch.softmax(ensemble_output, dim=1).cpu())
            all_targets.append(target)
            pred_a, pred_b, pred_c = ensemble_model(data, return_individual=True)
            individual_predictions.append({
                'model_a': torch.softmax(pred_a, dim=1).cpu(),
                'model_b': torch.softmax(pred_b, dim=1).cpu(),
                'model_c': torch.softmax(pred_c, dim=1).cpu()
            })
    
    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    df = pd.DataFrame(predictions, columns=["prob_site", "prob_non_site"])
    df["true_label"] = np.argmax(targets, axis=1)
    individual_a = torch.cat([pred['model_a'] for pred in individual_predictions]).numpy()
    individual_b = torch.cat([pred['model_b'] for pred in individual_predictions]).numpy()
    individual_c = torch.cat([pred['model_c'] for pred in individual_predictions]).numpy()
    
    df["model_a_prob_site"] = individual_a[:, 0]
    df["model_b_prob_site"] = individual_b[:, 0]
    df["model_c_prob_site"] = individual_c[:, 0]
    
    save_pred_path = f"results/pred_probs_{args.organism_name}_{args.splice_site_type}_ensemble.csv"
    os.makedirs(os.path.dirname(save_pred_path), exist_ok=True)
    df.to_csv(save_pred_path, index=False)
    print(f"Prediction probabilities saved to {save_pred_path}")

    ensemble_report = result_report(targets, predictions)
    
    report_a = result_report(targets, individual_a)
    report_b = result_report(targets, individual_b)  
    report_c = result_report(targets, individual_c)
    
    print(f"\nTest Results:")
    print(f"Model A: {report_a}")
    print(f"Model B: {report_b}")
    print(f"Model C: {report_c}")
    print(f"Ensemble: {ensemble_report}")
    
    return ensemble_report

def main():
    parser = argparse.ArgumentParser(description="Train SpliceDetector Ensemble Models")

    parser.add_argument("--organism_name", default=None, type=str, required=True, 
                       help="The input organism name.")
    parser.add_argument("--splice_site_type", default=None, type=str, choices=["donor", "acceptor"], 
                       required=True, help="Use this option to select donor or acceptor splice sites.")
    parser.add_argument("--test", action='store_true', default=False, help='Perform testing.')
    parser.add_argument("--train", action='store_true', default=False, help='Perform training and saving.')
    parser.add_argument("--num_train_epochs", default=60, type=int, 
                       help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, 
                       help="Batch size for training.")
    parser.add_argument("--learning_rate", default=0.001, type=float, 
                       help="The initial learning rate for AdamW.")
    parser.add_argument("--use_focal_loss", action='store_true', default=False, 
                       help='Use Focal Loss instead of CrossEntropy.')
    parser.add_argument("--verbose", default=1, type=int, choices=[0, 1], 
                       help='Display detailed information during training.')
    parser.add_argument("--report", action='store_true', default=False, 
                       help='Generate a report of the results.')

    args = parser.parse_args()

    train_dataset_path = f"data/encode_datasets/{args.organism_name}_{args.splice_site_type}_train.npz"
    val_dataset_path = f"data/encode_datasets/{args.organism_name}_{args.splice_site_type}_val.npz"
    test_dataset_path = f"data/encode_datasets/{args.organism_name}_{args.splice_site_type}_test.npz"
    if (not os.path.exists(train_dataset_path) or 
        not os.path.exists(val_dataset_path) or 
        not os.path.exists(test_dataset_path)):
        print("Encoding data...")
        splice_site_seq_path = f"data/dna_sequences/{args.organism_name}_{args.splice_site_type}_positive.txt"
        non_splice_site_seq_path = f"data/dna_sequences/{args.organism_name}_{args.splice_site_type}_negative.txt"
        x_train, x_val, x_test, y_train, y_val, y_test = encode_and_split_data(splice_site_seq_path, non_splice_site_seq_path)

        np.savez(train_dataset_path, x_train=x_train, y_train=y_train)
        np.savez(val_dataset_path, x_val=x_val, y_val=y_val)
        np.savez(test_dataset_path, x_test=x_test, y_test=y_test)
        print(f"Encoded datasets have been saved to 'data/encode_datasets/'")
    if args.train:
        print("="*60)
        print("Training SpliceDetector Ensemble Models")
        print("="*60)
        
        result_report_train = train_ensemble_models(args, train_dataset_path, val_dataset_path)
        
        if args.report:
            save_results_path = f"results/train_{args.organism_name}_{args.splice_site_type}_ensemble.txt"
            os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
            with open(save_results_path, 'wt') as f:
                f.write(result_report_train)
            print(f"Training results saved to {save_results_path}")
    if args.test:
        trained_model_path = f"models/trained_models/{args.organism_name}_{args.splice_site_type}_ensemble.pth"
        if not os.path.exists(trained_model_path):
            print(f"Ensemble model file {trained_model_path} does not exist. Please train the model first.")
            return
        
        print("="*60)
        print("Testing SpliceDetector Ensemble Model")
        print("="*60)
        
        result_report_test = predict_ensemble(args, test_dataset_path, trained_model_path)
        
        if args.report:
            save_results_path = f"results/test_{args.organism_name}_{args.splice_site_type}_ensemble.txt"
            os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
            with open(save_results_path, 'wt') as f:
                f.write(result_report_test)
            print(f"Test results saved to {save_results_path}")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()