SPECIES_DICT = {
    'species_1': [(7627, 7640)],
    'species_2': [(7642, 7653)],
    'species_3': [(7655, 7663)],
    'species_4': [(7665, 7679)],
    'species_5': [(7681, 7692), (7705, 7712)],
    'species_6': [(7694, 7703)],
    'species_7': [(7714, 7724), (7726, 7731), (7733, 7739), (7741, 7746)]
}


SPECIES_NAMES = {
    # Symphoricarpos albus
    'species_1': 'S. albus',
    # Lonicera xylosteum
    'species_2': 'L. xylosteum',
    # Crataegus monogyna
    'species_5': 'C. monogyna',
}


def get_species_number(im_name):
    im_digit = int(im_name.split('_')[1])

    for species_, ranges in SPECIES_DICT.items():
        for r in ranges:
            if (r[0] <= im_digit) and (im_digit <= r[1]):
                return species_
    
    raise ValueError(f'Image name {im_name} not found in species dict.')


def get_im_names(species_):
    im_names = []
    for im_name, ranges in SPECIES_DICT.items():
        if im_name == species_:
            for r in ranges:
                im_names += [f'IMG_{i}' for i in range(r[0], r[1]+1)]

    return im_names
