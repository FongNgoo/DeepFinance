#main.py
import torch
import random
import numpy as np
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig
import os

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

device = torch.device(
    "cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu"
)
set_seed(TrainConfig.seed)

def train_model(train_data, val_data=None):
    # --- 1. Setup Training Data ---
    # Move training data to device once
    s_o = train_data["s_o"].to(device)
    s_h = train_data["s_h"].to(device)
    s_c = train_data["s_c"].to(device)
    s_m = train_data["s_m"].to(device)
    s_n = train_data["s_n"].to(device)
    label = train_data["label"].to(device)

    # --- 2. Setup Validation Data (If provided) ---
    val_tensors = None
    if val_data:
        val_tensors = [
            val_data["s_o"].to(device), val_data["s_h"].to(device),
            val_data["s_c"].to(device), val_data["s_m"].to(device),
            val_data["s_n"].to(device), val_data["label"].to(device)
        ]

    # Initialize Model
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m.shape[-1],
        dim=TrainConfig.dim,
        input_dim=TrainConfig.input_dim,
        hidden_dim=TrainConfig.hidden_dim,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TrainConfig.learning_rate,
        weight_decay=TrainConfig.weight_decay
    )

    # Initialize trackers
    best_val_mcc = -1.0 # MCC ranges from -1 to 1, so start lower than 0
    best_val_acc = 0
    best_path = "output/best_model.pt"

    print(f"{'='*15}\n== TRAIN PHASE ==\n{'='*15}")

    for epoch in range(TrainConfig.epoch_num):
        # ============ TRAIN STEP ============
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (Loss calculation)
        loss = model(s_o, s_h, s_c, s_m, s_n, label, mode="train")
        
        loss.backward()
        
        # Check for NaN loss
        if not torch.isfinite(loss):
            print(f"❌ STOPPING: Loss is NaN at Epoch {epoch}")
            break
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ============ EVALUATION STEP ============
        model.eval()
        with torch.no_grad():
            # If we have validation data, use it. Otherwise, warn the user.
            if val_tensors:
                # Unpack validation tensors
                v_o, v_h, v_c, v_m, v_n, v_lbl = val_tensors
                acc, mcc = model(v_o, v_h, v_c, v_m, v_n, v_lbl, mode="test")
                eval_type = "VAL"
            else:
                # Fallback to training data (Not recommended for model selection)
                acc, mcc = model(s_o, s_h, s_c, s_m, s_n, label, mode="test")
                eval_type = "TRAIN (Warning: Overfitting)"

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"{eval_type} ACC {acc:.4f} | {eval_type} MCC {mcc:.4f}"
        )

        # ============ SAVE BEST MODEL ============
        # FIX: Update best_mcc properly
        if mcc > best_val_mcc:
            best_val_mcc = mcc  # <--- Updating the tracker (Crucial Fix)
            best_val_acc = acc  # <--- Save the ACC that corresponds to this MCC
    
    torch.save(model.state_dict(), best_path)
    print(f"   >>> ✅ Best Model Updated ({eval_type} MCC: {best_val_mcc:.4f})")

    print(f"\nFINAL RESULT → BEST MCC: {best_val_mcc:.4f}, ACC: {best_val_acc:.4f}")

def test_model(test_data):
    """
    Loads the best saved model state and evaluates it on the test set.
    """
    print(f"\n{'='*15}\n== TEST PHASE ==\n{'='*15}")
    
    # 1. Setup Test Data
    s_o = test_data["s_o"].to(device)
    s_h = test_data["s_h"].to(device)
    s_c = test_data["s_c"].to(device)
    s_m = test_data["s_m"].to(device)
    s_n = test_data["s_n"].to(device)
    label = test_data["label"].to(device)

    # 2. Re-initialize the Model Architecture
    # (Must match the architecture used during training exactly)
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m.shape[-1],
        dim=TrainConfig.dim,
        input_dim=TrainConfig.input_dim,
        hidden_dim=TrainConfig.hidden_dim,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        device=device
    ).to(device)

    # 3. Load the Best Weights
    best_path = "output/best_model.pt"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"✅ Loaded weights from: {best_path}")
    else:
        print(f"❌ Error: Model file not found at {best_path}")
        return

    # 4. Evaluation Loop
    model.eval()
    with torch.no_grad():
        acc, mcc = model(s_o, s_h, s_c, s_m, s_n, label, mode="test")

    # 5. Report Results
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   -------------------")
    print(f"   Accuracy (ACC): {acc:.4f}")
    print(f"   MCC Score     : {mcc:.4f}")
    print(f"   -------------------")



if __name__ == "__main__":
    dp = data_prepare(r"D:\Project\NCKH\data\env_data\semi_final_data.pkl")

    train_data, test_data = dp.prepare_data(
        stock_name="TSLA",
        window_size=20,
        future_days=1,
        train_ratio=0.8
    )

    train_model(train_data)
    test_model(test_data)
