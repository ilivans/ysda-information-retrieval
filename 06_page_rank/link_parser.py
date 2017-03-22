import os
import re
from HTMLParser import HTMLParser
from urllib import urlopen
from urlparse import urljoin


class LinkParser(HTMLParser):
    _BASE_URL = "https://simple.wikipedia.org/wiki/"
    _BASE_URL_LEN = len(_BASE_URL)
    _INVALID_NAME_PATTERN = re.compile(
        "((Category|File|Help|Media|Special|Talk|Template|Template_talk|User|User_talk|Wikipedia):|#)")
    _PAGES_DIR = "/media/ssd/simple.wiki/pages"
    if not os.path.exists(_PAGES_DIR):
        os.mkdir(_PAGES_DIR)

    @staticmethod
    def valid_url(url):
        prefix, name = url[:LinkParser._BASE_URL_LEN], url[LinkParser._BASE_URL_LEN:]
        if prefix != LinkParser._BASE_URL:
            return False
        return LinkParser._INVALID_NAME_PATTERN.match(name) is None

    @staticmethod
    def normalize_url(url):
        hashtag_pos = url.find("#", LinkParser._BASE_URL_LEN)
        if hashtag_pos > -1:
            url = url[:hashtag_pos]
        slash_pos = url.find("/", LinkParser._BASE_URL_LEN)
        if slash_pos > -1:
            url = url[:slash_pos]
        return url

    @staticmethod
    def extract_name(url):
        return url[LinkParser._BASE_URL_LEN:]

    @staticmethod
    def drop_page(url, page):
        dump_path = os.path.join(LinkParser._PAGES_DIR, LinkParser.extract_name(url))
        if not os.path.exists(dump_path):
            with open(dump_path, "w") as f:
                f.write(page)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    new_url = urljoin(LinkParser._BASE_URL, value)
                    if LinkParser.valid_url(new_url):
                        self.links.add(LinkParser.normalize_url(new_url))

    def get_links(self, url):
        """
        Returns filtered and normalized links from the page retrieved by `url`.
        Also dumps loaded page.
        :param url: url of the page
        :return: links found on the page
        """
        self.links = set()
        response = urlopen(url)
        if response.info()['Content-Type'].startswith('text/html'):
            content = response.read()
            LinkParser.drop_page(url, content)
            self.feed(content.decode("utf-8"))
            return self.links
        else:
            return set()
