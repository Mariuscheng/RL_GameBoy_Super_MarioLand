# DQN Super Mario Land

這個項目使用深度Q網路 (DQN) 訓練AI代理玩Super Mario Land遊戲。使用PyBoy模擬器、Gymnasium強化學習框架和PyTorch神經網路庫。

## 功能特點

- 使用CNN架構的DQN代理
- 自定義的Mario環境，基於PyBoy和Gymnasium
- 經驗回放記憶體
- ε-貪婪策略進行動作選擇
- 目標網路更新

## 依賴項

- Python 3.8+
- PyTorch
- Gymnasium
- PyBoy
- NumPy

## 安裝

1. 安裝Python依賴項：

   ```bash
   pip install torch gymnasium pyboy numpy
   ```
2. 確保有Super Mario Land ROM文件 (`rom.gb`) 在項目根目錄。

## 使用方法

運行訓練腳本：

```bash
python main.py
```

腳本將開始訓練DQN代理玩Super Mario Land。

## 環境描述

### 觀察空間 (Observation Space)

觀察空間是一個形狀為 (17,) 的numpy array，dtype=float32，包含遊戲狀態、Mario狀態和敵人狀態。

| 索引 | 觀察項目           | 描述                  | 來源                | 最小值 | 最大值 |
| ---- | ------------------ | --------------------- | ------------------- | ------ | ------ |
| 0    | lives_left         | Mario剩餘生命數       | PyBoy MarioLand API | 0      | 99     |
| 1    | coins              | 收集的金幣數          | PyBoy MarioLand API | 0      | 99     |
| 2    | level_progress     | 關卡進度（全局X位置） | PyBoy MarioLand API | 0      | 2601   |
| 3    | score              | 遊戲分數              | PyBoy MarioLand API | 0      | 999999 |
| 4    | time_left          | 剩餘時間（秒）        | PyBoy MarioLand API | 0      | 400    |
| 5    | world              | 當前世界              | PyBoy MarioLand API | 0      | 4      |
| 6    | stage              | 當前關卡              | PyBoy MarioLand API | 0      | 3      |
| 7    | mario_x            | Mario X位置           | RAM 0xC202          | 0      | 81     |
| 8    | mario_y            | Mario Y位置           | RAM 0xC201          | 0      | 134    |
| 9    | power_status       | 力量狀態              | RAM 0xFF99          | 0      | 4      |
| 10   | enemy_status       | 敵人狀態              | RAM 0xD100          | 0      | 255    |
| 11   | player_status      | 玩家狀態              | RAM 0xD100          | 0      | 2      |
| 12   | superball_status   | 超級球狀態            | RAM 0xFF99          | 0      | 2      |
| 13   | is_died            | 死亡標誌              | RAM 0xFFA6          | 0      | 1      |
| 14   | stage_over         | 關卡結束標誌          | level_progress      | 0      | 1      |
| 15   | ground_flag        | 地面標誌              | RAM 0xC20A          | 0      | 1      |
| 16   | power_status_timer | 力量狀態計時器        | RAM 0xFFA6          | 0      | 2      |

### 動作空間 (Action Space)

動作空間是離散的，有13個可能動作：

| 動作值 | 動作名稱    | 描述              |
| ------ | ----------- | ----------------- |
| 0      | NOOP        | 無操作            |
| 1      | LEFT        | 向左移動          |
| 2      | RIGHT       | 向右移動          |
| 3      | UP          | 向上移動          |
| 4      | JUMP        | 跳躍 (A鍵)        |
| 5      | FIRE        | 發射 (B鍵)        |
| 6      | LEFT_PRESS  | 按左鍵            |
| 7      | RIGHT_PRESS | 按右鍵            |
| 8      | JUMP_PRESS  | 按跳躍鍵          |
| 9      | LEFT_RUN    | 跑步向左 (B + 左) |
| 10     | RIGHT_RUN   | 跑步向右 (B + 右) |
| 11     | LEFT_JUMP   | 跳躍向左 (A + 左) |
| 12     | RIGHT_JUMP  | 跳躍向右 (A + 右) |

### 獎勵系統 (Reward System)

| 條件                                     | 獎勵         |
| ---------------------------------------- | ------------ |
| 遊戲結束 (pyboy.memory[0xC0A4] == 0x39)  | -100         |
| 死亡/重生 (pyboy.memory[0xFFA6] == 0x90) | -100         |
| 關卡進度增加                             | +進度差值    |
| 收集金幣                                 | +金幣數 * 10 |
| 獲得力量 (從小變大)                      | +100         |
| 每步時間懲罰                             | -0.1         |

### 終止條件 (Termination Conditions)

- **terminated**: 生命值為0或時間為0時重置遊戲
- **truncated**: 進度達到2601（通關）時停止遊戲

## DQN架構

- **輸入**: (batch, 17)
- **FC1**: 128個神經元
- **FC2**: 128個神經元
- **FC3**: 13個輸出 (動作數)

## 訓練參數

- **BATCH_SIZE**: 64
- **GAMMA**: 0.99
- **EPS_START**: 1.0
- **EPS_END**: 0.1
- **EPS_DECAY**: 500000
- **TAU**: 0.001
- **LR**: 1e-4
- **MEMORY_SIZE**: 100000

## 參考資料

- [Gymnasium](https://gymnasium.farama.org/)
- [PyBoy Documentation](https://docs.pyboy.dk/)
- [PyBoy Mario Land API](https://docs.pyboy.dk/plugins/game_wrapper_super_mario_land.html)
- [Super Mario Land RAM Map](https://datacrystal.tcrf.net/wiki/Super_Mario_Land/RAM_map)
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
