from unittest import TestCase
from link_parser import LinkParser

valid_url = LinkParser.valid_url
normalize_url = LinkParser.normalize_url
extract_name = LinkParser.extract_name


class Tests(TestCase):
    def test_valid_url(self):
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/Main_Page"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/Weather"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/Blu-ray"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/Family_(biology)"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/BAFTA_Academy_Fellowship_Award#cite_note-off-6"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/A.S._Fortis_Trani"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/Janet.#mw-head"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/Ender%27s_Game"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/ISO_3166-2:BR"))
        self.assertTrue(valid_url("https://simple.wikipedia.org/wiki/List_of_record_labels:_I%E2%80%93Q"))

        self.assertFalse(valid_url("https://commons.wikimedia.org/wiki/Main_Page"))
        self.assertFalse(
            valid_url("https://simple.wikipedia.org/w/index.php?title=Special:UserLogin&returnto=Main+Page"))
        self.assertFalse(valid_url("http://en.wikiversity.org/?uselang=mk"))

        self.assertFalse(valid_url("https://simple.wikipedia.org/wiki/"))
        self.assertFalse(valid_url("https://simple.wikipedia.org/wiki/Special:RecentChangesLinked/Summer"))
        self.assertFalse(valid_url("https://simple.wikipedia.org/wiki/File:Science-symbol-2.svg"))
        self.assertFalse(valid_url("https://simple.wikipedia.org/wiki/Category:All_articles_with_dead_external_links"))
        self.assertFalse(valid_url("https://simple.wikipedia.org/wiki/Wikipedia:Simple_start"))

    def test_normalize_url(self):
        self.assertEqual(normalize_url("https://simple.wikipedia.org/wiki/Main_Page"),
                                       "https://simple.wikipedia.org/wiki/Main_Page")
        self.assertEqual(normalize_url("https://simple.wikipedia.org/wiki/Main_Page/Something"),
                                       "https://simple.wikipedia.org/wiki/Main_Page")
        self.assertEqual(normalize_url("https://simple.wikipedia.org/wiki/Janet.#mw-head"),
                                       "https://simple.wikipedia.org/wiki/Janet.")

    def test_extract_name(self):
        self.assertEqual(extract_name("https://simple.wikipedia.org/wiki/Main_Page"), "Main_Page")
