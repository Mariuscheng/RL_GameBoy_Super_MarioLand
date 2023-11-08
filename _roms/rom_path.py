import os


# a dictionary mapping ROM paths first by lost levels, then by ROM hack mode
_ROM_PATHS = {
    # the dictionary of lost level ROM paths
    True: {
        'vanilla': 'SuperMarioLand_Gameboy.gb',
    },
    # the dictionary of Super Mario Bros. 1 ROM paths
    '''''''''
    False: {
        'vanilla': 'SuperMarioLand_Gameboy.gb',
    }
    ''''''''''
}


def rom_path(lost_levels, rom_mode):
    """
    Return the ROM filename for a game and ROM mode.

    Args:
        lost_levels (bool): whether to use the lost levels ROM
        rom_mode (str): the mode of the ROM hack to use as one of:
            - 'vanilla'
            - 'pixel'
            - 'downsample'
            - 'vanilla'

    Returns (str):
        the ROM path based on the input parameters

    """
    # Type and value check the lost levels parameter
    if not isinstance(lost_levels, bool):
        raise TypeError('lost_levels must be of type: bool')
    # try the unwrap the ROM path from the dictionary
    try:
        rom = _ROM_PATHS[lost_levels][rom_mode]
    except KeyError:
        raise ValueError('rom_mode ({}) not supported!'.format(rom_mode))
    # get the absolute path for the ROM
    rom = os.path.join(os.path.dirname(os.path.abspath(__file__)), rom)

    return rom


# explicitly define the outward facing API of this module
__all__ = [rom_path.__name__]