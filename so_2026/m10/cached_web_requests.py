#! /usr/bin/env python

# Source - https://stackoverflow.com/q/80010011
# Posted by Jeyekomon, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-15, License - CC BY-SA 4.0

import logging
from http.cookiejar import Cookie, CookiePolicy
from typing import TYPE_CHECKING
from urllib.request import Request

from beartype import beartype
from requests.cookies import MockRequest
from requests_cache import CachedSession

if TYPE_CHECKING:
    from urllib.request import Request

    from requests.cookies import MockRequest

logging.basicConfig(level="INFO")

uniq = {
    "wmf-uniq": "ZWuWfK7fKegG6hO0s8JYlgP5AAEBAFvdab-KTV9nESii8Pk1O-TFSHMuaL2Pxl1o",
    "wmf-dp": "3c2,61b,4ee,bca",
}


@beartype
class BlockAll(CookiePolicy):
    netscape = True
    rfc2965 = hide_cookie2 = False

    def set_ok(self, cookie: Cookie, request: MockRequest | Request) -> bool:
        assert self
        assert cookie
        assert request
        return False

    def return_ok(self, cookie: Cookie, request: Request) -> bool:
        assert self
        assert cookie
        assert request
        return False

    def domain_return_ok(self, domain: str, request: Request) -> bool:
        assert self
        assert domain
        assert request
        return False

    def path_return_ok(self, path: str, request: Request) -> bool:
        assert self
        assert path
        assert request
        return False


codes = ["Q41", "Q712", "Q96", "Q115", "Q902"]


def main() -> None:
    session = CachedSession()
    session.cookies.set_policy(BlockAll())
    cache_size = session.cache.count()
    from_cache = 0

    for code in codes:
        response = session.get(
            f"https://www.wikidata.org/wiki/{code}",
            headers={"user-agent": "Wikidata scraper"},
        )
        if response.from_cache:
            from_cache += 1

    print("requests:  ", len(codes))
    print("from cache:", from_cache)
    print("stored:    ", session.cache.count() - cache_size)


if __name__ == "__main__":
    main()
