# GameBoy Super Mario land (A.I improve)

## Observation Space (9,)


| Num | Observation                               | Description                                                                                                                          | Source                   | Min | Max    |
| :---- | :------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----- | :------- |
| 0   | lives_left                                | The number of lives Mario has left.                                                                                                  | PyBoy MarioLand API      | 0   | 99     |
| 1   | socre                                     | The score provided by the game                                                                                                       | PyBoy MarioLand API      | 0   | 999999 |
| 2   | time_left                                 | The number of seconds left to finish the level.                                                                                      | PyBoy MarioLand API      | 0   | 400    |
| 3   | level_progress                            | An integer of the current "global" X position in this level. Can be used for AI scoring.                                             | PyBoy MarioLand API      | 250 | 2601   |
| 4   | coins                                     | The number of collected coins.                                                                                                       | PyBoy MarioLand API      | 0   | 99     |
| 5   | World                                     | Provides the current "world" Mario is in, as a tuple of as two integers (world, level).                                              | PyBoy MarioLand API      | 1   | 4      |
| 6   | Stage                                     | Provides the current "world" Mario is in, as a tuple of as two integers (world, level). Python code : World[1]                       | PyBoy MarioLand API      | 1   | 3      |
| 7   | mario x position (pyboy.memory[0xC202])   | Mario's X position relative to the screen                                                                                            | Super Mario Land/RAM map | 0   | 81     |
| 8   | mario y position (pyboy.memory[0xC201])   | Mario's Y position relative to the screen                                                                                            | Super Mario Land/RAM map | 0   | 134    |
| 9   | Game Over (pyboy.memory[0xC0A4])          | ? (0x39 = Game Over)                                                                                                                 | Super Mario Land/RAM map | 0   | 57     |
| 10  | Power State (pyboy.memory[0xFF99])        | Powerup Status (0x00 = small, 0x01 = growing, 0x02 = big with or without superball, 0x03 = shrinking, 0x04 = invincibility blinking) | Super Mario Land/RAM map | 0   | 4      |
| 11  | Mario jump routine (pyboy.memory[0xC207]) | Probably used in Mario's jump routine. (0x00 = Not jumping, 0x01 = Ascending, 0x02 = Descending)                                     | Super Mario Land/RAM map | 0   | 2      |

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

## terminated(運行mario.reset_game())


| Item              | Terminated Condition |
| :------------------ | ---------------------- |
| if lives_left = 0 | mario.reset_game()   |
| if time_left = 0  | mario.reset_game()   |

## terminated(運行pyboy.stop() )


| Item           | Terminated Condition |
| :--------------- | ---------------------- |
| level_progress | = 2601 (Goal)        |

## truncated


| Item     | Truncated Condition |
| :--------- | --------------------- |
| 最大步數 | = 2601              |

## Sample

1. Initial observation: [  2.   0. 400. 251.   0.   1.   1.  50. 134.   0.   0.   0.]
2. Observation space: 12
3. Action space: 13

## Reference

- [Gym Retro](https://gymnasium.farama.org/)
- [PyBoy API](https://docs.pyboy.dk/index.html)
- [PyBoy Mario Land API](https://docs.pyboy.dk/plugins/game_wrapper_super_mario_land.html)
- [Super Mario Land Ram Map Wiki](https://datacrystal.tcrf.net/wiki/Super_Mario_Land/RAM_map)
