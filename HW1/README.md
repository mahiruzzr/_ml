# Hill Climbing 旅行推銷員問題 (TSP) 求解器

這是一個使用 **Hill Climbing (爬山演算法)** 來求解旅行推銷員問題 (TSP) 的簡單 C++ 實作。

## 專案說明

本程式透過簡單的爬山法（Hill Climbing）來尋找較短的旅行路線。程式會不斷嘗試交換兩個城市的位置，如果新路線總距離較短（或相等），則接受該解，直到達到停止條件為止。

### 主要特點
- 使用 `std::vector<int>` 表示城市訪問順序
- 鄰居產生方式：隨機交換兩個不同城市的位置
- 採用「負距離」技巧，讓原本的最大化 hill climbing 框架可以求最小化問題
- 支援自訂城市數量與距離矩陣
- 簡單易懂，適合學習 metaheuristic 演算法

## 編譯與執行

```bash
# 編譯
g++ hillClimbing.cpp -o tsp_hillclimbing -std=c++11

# 執行
./tsp_hillclimbing
