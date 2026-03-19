import unittest

from src.inference.wikidata_properties import map_relation_to_properties


class WikidataPropertiesTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(map_relation_to_properties("author"), ["P50"])
        self.assertEqual(map_relation_to_properties("publication_year"), ["P577"])
        self.assertEqual(map_relation_to_properties("unknown_relation"), [])


if __name__ == "__main__":
    unittest.main()
