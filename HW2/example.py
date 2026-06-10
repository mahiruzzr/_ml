import random
from nn0 import Value, Adam, linear, cross_entropy

def main():
    # 設定隨機種子以確保結果可重現
    random.seed(42)

    ## 1. 準備訓練數據 (二維輸入 -> 二分類標籤)
    # 規則：如果 x0 + x1 > 0 則標籤為 1，否則為 0
    raw_data = [
        ([1.2, 0.5], 1),
        ([-0.8, -1.1], 0),
        ([0.5, 1.5], 1),
        ([-1.5, 0.2], 0),
        ([2.0, -0.5], 1),
        ([-0.2, -2.0], 0),
    ]
    
    ## 2. 初始化模型參數 (權重矩陣 W)
    # 輸入維度 2，輸出維度 2 (分別代表類別 0 和類別 1 的 Logits)
    # W 的形狀為 (2, 2)
    W = [
        [Value(random.uniform(-0.5, 0.5)), Value(random.uniform(-0.5, 0.5))], # 類別 0 的權重
        [Value(random.uniform(-0.5, 0.5)), Value(random.uniform(-0.5, 0.5))]  # 類別 1 的權重
    ]
    
    # 將所有需要更新的參數攤平送給優化器
    params = [w_ij for row in W for w_ij in row]
    optimizer = Adam(params, lr=0.2)
    
    print("--- 開始訓練 ---")
    num_steps = 40
    
    for step in range(num_steps):
        total_loss = 0
        
        # 對每個樣本進行前向傳播與反向傳播
        for x_raw, target_idx in raw_data:
            # 將輸入包裝成 Value 節點
            x = [Value(v) for v in x_raw]
            
            # 前向傳播 (Forward Pass)
            # logits = W @ x -> 輸出兩個類別的得分
            logits = linear(x, W)
            
            # 計算數值穩定的 Cross-Entropy Loss
            loss = cross_entropy(logits, target_idx)
            total_loss += loss.data
            
            # 反向傳播 (Backward Pass)
            # 計算當前計算圖中所有節點的梯度 (grad)
            loss.backward()
            
            # 使用 Adam 優化器更新參數，並自動清空梯度
            # 這裡模擬學習率隨步數線性衰減
            lr_t = optimizer.lr * (1 - step / num_steps)
            optimizer.step(lr_override=lr_t)
            
        # 每 10 個 Epoch 觀察一次 Loss 的變化
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1:02d} | 平均 Loss: {total_loss / len(raw_data):.4f}")
            
    print("\n--- 訓練完成，測試模型預測能力 ---")
    test_points = [[1.5, 1.0], [-1.0, -1.0]]
    for tp in test_points:
        x_test = [Value(v) for v in tp]
        logits = linear(x_test, W)
        # 找出數值最大的索引作為預測結果
        prediction = 0 if logits[0].data > logits[1].data else 1
        print(f"輸入點: {tp} -> 模型預測類別: {prediction} (Logits: [{logits[0].data:.2f}, {logits[1].data:.2f}])")

if __name__ == "__main__":
    main()
