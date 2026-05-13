#include <iostream>
#include <string>
#include <cstdlib>   // rand, srand
#include <ctime>     // time
#include <cmath>

class Solution {
private:
    int x;

public:
    // 建構子
    Solution(double val = 0.0) : x(val) {}

    // 1. 生成鄰居解（nheight = neighbor height）
    Solution nheight() const {
        Solution neighbor = *this;           // 複製目前解
        
        // 隨機擾動（可調整步長）
        double step = 0.5 + (rand() % 100) / 100.0; // 0.5 ~ 1.5 的隨機步長
        if (rand() % 2 == 0) {
            neighbor.x += step;
        } else {
            neighbor.x -= step;
        }
        
        return neighbor;
    }
    // 2. 計算高度（適應度 / 目標函數值）
    double height() const {
        // 目標函數：f(x) = -(x-3.5)^2 + 12.25
        return -(x - 3.5) * (x - 3.5) + 12.25;
    }
    // 3. 轉成字串（方便輸出觀察）
    std::string str() const {
        char buf[100];
        snprintf(buf, sizeof(buf), "x = %.4f, height = %.4f", x, height());
        return std::string(buf);
    }
};

int hillClimbing(Solution s, int MaxGen, int MaxFails){
    std::cout << s.str() << endl;
    int fails = 0;
    for (int i=0;i<MaxGen;i++){
        Solution snew = s.nheight();
        int sheight = s.height();
        int nheight = snew.height();
        if(nheight >= sheight){
            cout << i << ":" << snew.str() << endl;
            s = snew;
            fails = 0;  
        }else{
            fails +=1;
            if(fails >= MaxFails){
                break;
            }
        }
    }cout << "Solution: " << s.str() << endl;
   return s;
} 