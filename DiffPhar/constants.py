import numpy as np
from rdkit import Chem
import torch

# ------------------------------------------------------------------------------
# Computational
# ------------------------------------------------------------------------------
FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

# ------------------------------------------------------------------------------
# Bond parameters
# ------------------------------------------------------------------------------

# margin1, margin2, margin3 = 10, 5, 3
margin1, margin2, margin3 = 3, 2, 1

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186, 'C': 160}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

# https://en.wikipedia.org/wiki/Covalent_radius#Radii_for_multiple_bonds
# (2022/08/14)
covalent_radii = {'H': 32, 'C': 60, 'N': 54, 'O': 53, 'F': 53, 'B': 73,
                  'Al': 111, 'Si': 102, 'P': 94, 'S': 94, 'Cl': 93, 'As': 106,
                  'Br': 109, 'I': 125, 'Hg': 133, 'Bi': 135}

# ------------------------------------------------------------------------------
# Backbone geometry
# Taken from: Bhagavan, N. V., and C. E. Ha.
# "Chapter 4-Three-dimensional structure of proteins and disorders of protein misfolding."
# Essentials of Medical Biochemistry (2015): 31-51.
# https://www.sciencedirect.com/science/article/pii/B978012416687500004X
# ------------------------------------------------------------------------------
N_CA_DIST = 1.47
CA_C_DIST = 1.53
N_CA_C_ANGLE = 110 * np.pi / 180


# ------------------------------------------------------------------------------
# Dataset-specific constants
# ------------------------------------------------------------------------------
dataset_params = {}

dataset_params['crossdock_full'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others'],
      'phar_encoder': {'Aromatic': 0, 'Hydrophobe': 1, 'PosIonizable': 2,'NegIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6, 'others': 7},
      'phar_decoder': ['Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable', 'Acceptor', 'Donor', 'LumpedHydrophobe', 'others'],
      'aa_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9, 'others': 10},
      'aa_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F', 'others'],
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF', '#ffb5b5'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'phar_hist': {'Aromatic': 176393, 'Hydrophobe': 329938, 'PosIonizable': 38876, 'NegIonizable': 28234, 'Acceptor': 485363, 'Donor': 303290,
                    'LumpedHydrophobe': 124515, 'others': 30892},
      'aa_hist': {'C': 23481798, 'N': 6139100, 'O': 6753114, 'S': 278864, 'B': 0, 'Br': 0, 'Cl': 0, 'P': 0, 'I': 0, 'F': 0, 'others': 0},
}

dataset_params['crossdock'] = {
      'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9},
      'atom_decoder': ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F'],
      'phar_encoder': {'Aromatic': 0, 'Hydrophobe': 1, 'PosIonizable': 2, 'NegIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6, 'others': 7},
      'phar_decoder': ['Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable', 'Acceptor', 'Donor', 'LumpedHydrophobe','others'],
      'aa_encoder': {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19},
      'aa_decoder': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
      # PyMOL colors, see: https://pymolwiki.org/index.php/Color_Values#Chemical_element_colours
      'colors_dic': ['#33ff33', '#3333ff', '#ff4d4d', '#e6c540', '#ffb5b5', '#A62929', '#1FF01F', '#ff8000', '#940094', '#B3FFFF'],
      'radius_dic': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
      'phar_hist': {'Aromatic': 176393, 'Hydrophobe': 329938, 'PosIonizable': 38876, 'NegIonizable': 28234, 'Acceptor': 485363, 'Donor': 303290,
                    'LumpedHydrophobe': 124515, 'others': 30892},
      'aa_hist': {'A': 277175, 'C': 92406, 'D': 254046, 'E': 201833, 'F': 234995, 'G': 376966, 'H': 147704, 'I': 290683, 'K': 173210, 'L': 421883, 'M': 157813, 'N': 174241, 'P': 148581, 'Q': 120232, 'R': 173848, 'S': 274430, 'T': 247605, 'V': 326134, 'W': 88552, 'Y': 226668},
}
