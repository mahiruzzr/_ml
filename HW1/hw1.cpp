#include <iostream>
#include <string>
#include <cstdlib>   
#include <ctime>     
#include <cmath>
#include <vector>
#include <algorithm>

class Solution {
private:
    std::vector<int> tour;     // 改成路徑（排列）

    // 城市距離矩陣（範例：6 個城市）
    static const int N = 6;
    static double dist[N][N];

public:
    // 建構子：產生初始解（0->1->2...->N-1）
    Solution() {
        tour.resize(N);
        for(int i = 0; i < N; i++) tour[i] = i;
        // 隨機打亂產生初始解
        std::random_shuffle(tour.begin(), tour.end());
    }

    // 1. 生成鄰居解（最小修改版：隨機交換兩個城市）
    Solution nheight() const {
        Solution neighbor = *this;
        
        int i = rand() % N;
        int j = rand() % N;
        while(i == j) j = rand() % N;
        
        std::swap(neighbor.tour[i], neighbor.tour[j]);
        
        return neighbor;
    }

    // 2. 計算高度（因為原本是最大化，所以回傳「負的總距離」）
    double height() const {
        double total = 0.0;
        for(int i = 0; i < N; i++) {
            int from = tour[i];
            int to   = tour[(i+1) % N];
            total += dist[from][to];
        }
        return -total;          // 負距離 → 越短越好（高度越高）
    }

    // 3. 轉成字串
    std::string str() const {
        std::string s = "Tour: ";
        for(int city : tour) {
            s += std::to_string(city) + " ";
        }
        s += "→ " + std::to_string(tour[0]);
        s += "  Length = " + std::to_string(-height());  // 顯示實際距離
        return s;
    }

    // 距離矩陣初始化（對稱 TSP）
    static void initDist() {
        // 範例距離矩陣（可自行修改）
        double d[N][N] = {
            {0, 10, 15, 20, 25, 30},
            {10, 0, 35, 25, 30, 15},
            {15, 35, 0, 30, 20, 25},
            {20, 25, 30, 0, 15, 35},
            {25, 30, 20, 15, 0, 10},
            {30, 15, 25, 35, 10, 0}
        };
        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
                dist[i][j] = d[i][j];
    }
};

// 靜態成員定義
double Solution::dist[Solution::N][Solution::N];

// ====================== hillClimbing ======================
int hillClimbing(Solution s, int MaxGen, int MaxFails){
    std::cout << "Initial: " << s.str() << std::endl;
    
    int fails = 0;
    for (int i = 0; i < MaxGen; i++){
        Solution snew = s.nheight();
        double sheight = s.height();
        double nheight = snew.height();
        
        if(nheight >= sheight){                    // 接受更好的解
            std::cout << i << ": " << snew.str() << std::endl;
            s = snew;
            fails = 0;  
        } else {
            fails += 1;
            if(fails >= MaxFails){
                std::cout << "=== Stop due to too many fails ===\n";
                break;
            }
        }
    }
    std::cout << "Final Solution: " << s.str() << std::endl;
    return 0;   // 原本 return s 但型別不符，這裡簡化
}

// ====================== main ======================
int main() {
    srand(time(nullptr));
    Solution::initDist();        // 初始化距離矩陣
    
    Solution initial;
    hillClimbing(initial, 5000, 50);   // 可調整參數
    
    return 0;
}
