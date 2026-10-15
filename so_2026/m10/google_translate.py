#! /usr/bin/env python

# Source - https://stackoverflow.com/q/80010106
# Posted by rmor23
# Retrieved 2026-10-15, License - CC BY-SA 4.0

import asyncio
from pprint import pp

from googletrans import Translator


async def translate_text() -> None:
    async with Translator() as translator:
        result = await translator.translate("Hola mundo.", src="es", dest="en")
        print("Googletrans tutorial result:")
        print(result)
        pp(result.extra_data)
        print(result.text)
        assert result.text == "Hello world."


if __name__ == "__main__":
    asyncio.run(translate_text())
