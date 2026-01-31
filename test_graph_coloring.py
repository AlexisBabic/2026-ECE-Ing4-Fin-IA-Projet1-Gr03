"""
Suite de tests unitaires pour le projet de coloration de graphe.
Objectifs :
1. Vérifier l'intégrité de la structure de données (Graphe).
2. Valider l'algorithme de coloration (respect des contraintes de couleurs et d'adjacence).
"""

import unittest
from src.graph_coloring import create_sample_graph, solve_graph_coloring


class TestGraphColoring(unittest.TestCase):

    def test_sample_graph_creation(self):
        """Vérifie que le graphe exemple est correctement instancié."""
        G = create_sample_graph()
        self.assertEqual(len(G.nodes()), 6)  # 5 + center
        self.assertEqual(len(G.edges()), 10)  # pentagon + 5 to center

    def test_coloring_solution(self):
        """Vérifie que la solution retournée respecte toutes les règles de coloration de graphe."""
        G = create_sample_graph()
        coloring, num_colors = solve_graph_coloring(G, max_colors=4)
        self.assertIsNotNone(coloring)
        self.assertGreaterEqual(num_colors, 1)
        self.assertLessEqual(num_colors, 4)

        # Vérifie qu'aucun nœud adjacent ne partage la même couleur.
        for u, v in G.edges():
            self.assertNotEqual(coloring[u], coloring[v])


if __name__ == '__main__':

    unittest.main()



