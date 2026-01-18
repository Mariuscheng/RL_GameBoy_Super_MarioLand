# GameBoy Super Mario land (A.I improve)

## Observation Space (9,)


| Num | Observation                               | Min | Max    |
| :---- | :------------------------------------------ | ----- | :------- |
| 0   | lives_left                                | 0   | 99     |
| 1   | socre                                     | 0   | 999999 |
| 2   | time_left                                 | 0   | 400    |
| 3   | level_progress                            | 250 | 2601   |
| 4   | coins                                     | 0   | 99     |
| 5   | World                                     | 1   | 4      |
| 6   | Stage                                     | 1   | 3      |
| 7   | mario x position (pyboy.memory[0xC202])   | 0   | 81     |
| 8   | mario y position (pyboy.memory[0xC201])   | 0   | 134    |
| 9   | Game Over (pyboy.memory[0xC0A4])          | 0   | 57     |
| 10  | Power State (pyboy.memory[0xFF99])        | 0   | 4      |
| 11  | Mario jump routine (pyboy.memory[0xC207]) | 0   | 2      |

## Action Space (GameBoy)


| Button               | Action                           |
| :--------------------- | ---------------------------------- |
| Up                   | Up                               |
| Down                 | Down                             |
| Left                 | Left                             |
| Right                | Right                            |
| Fire                 | B                                |
| Jump                 | A                                |
| Run                  | B(LongPress) + Right/Left        |
| Right-Jump/Left-Jump | B(LongPress) + Right/Left + Jump |

## Reward


| Item                  | Reward System(One Time) |
| :---------------------- | ------------------------- |
| if lives_left - 1     | - 50                    |
| if level_progress + 1 | + 1                     |
| if lives_left = 0     | - 100                   |
| Power_state + 1       | + 100                   |
| Power_state = 0       | - 100                   |

## terminated


| Item              | Terminated Condition |
| :------------------ | ---------------------- |
| level_progress    | = 2601 (Goal)        |
| if lives_left = 0 | mario.reset_game()   |
| if time_left = 0  | mario.reset_game()   |

## truncated


| Item     | Truncated Condition |
| :--------- | --------------------- |
| 最大步數 | = 2601              |

## Reference

- [Gym Retro](https://gymnasium.farama.org/)
- [PyBoy API](https://docs.pyboy.dk/index.html)
- [PyBoy Mario Land API](https://docs.pyboy.dk/plugins/game_wrapper_super_mario_land.html)
- [Super Mario Land Ram Map Wiki](https://datacrystal.tcrf.net/wiki/Super_Mario_Land/RAM_map)
