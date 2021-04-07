# ENAS-Keras
Implementation of Efficient Neural Architecture Search with Keras

en modifiant blocks_array l. 72 (Cell.py), essayer d'autres combinaisons de blocks

	par exemple créer 2 blocks qui se suivent (donc 2 même inputs / outputs)


Ensuite, il faut alterner encore 1 reducCell puis 1 autre convCell puis encore un reduc

	=> il faut un all_layers (dans Block.py) different pour chaque nouvelle Cell

			=> aller au plus simple pour l'instant, mettre une condition


tester tous les types de convCell en 3 blocks possibles

faire des reducCell PUTAIN T'A CAPTÉ


Faire une fonction qui gènère aléatoire des network dans cet espace !
