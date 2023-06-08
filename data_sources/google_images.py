import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import quote_plus

from chisel.data_sources.base_data_source import BaseDataSource
from chisel.storage.local_fs import LocalFS


class GoogleImages(BaseDataSource):
    def __init__(self, save_html: bool = False):
        super().__init__()
        self.save_html = save_html
        self.search_url = "https://www.google.com/search?q={search_query}&tbm=isch"
        self.download_dir = Path(".")

    def _save_data(self, image_url, image_page_url):
        image_name = image_url.split("/")[-1]
        self.storage.download_img_from_url(image_url, filename=image_name, ext=".png")
        if self.save_html:
            self._save_html(image_page_url, image_name)

    def _save_html(self, image_page_url: str, image_name: str):
        # Save the web page
        # self.storage.
        # with open(page_path, "w", encoding="utf-8") as file:
        #     file.write(requests.get(image_page_url).text)
        pass

    def set_download_dir(self, download_dir: Path) -> None:
        self.download_dir = download_dir
        self.storage = LocalFS(Path("~/chisel") / download_dir)

    # Function to search Google Images for a given phrase and download the images
    def search_and_download_images(
        self,
        search_phrase: str,
        num_images: int,
    ):
        # Prepare the search URL
        search_query = quote_plus(search_phrase)

        full_url = self.search_url.format(search_query=search_query)
        response = requests.get(full_url)

        # Create a BeautifulSoup object from the response content
        soup = BeautifulSoup(response.content, "html.parser")
        image_elements = soup.find_all("img")

        # Download and save each image
        for i, image_element in enumerate(image_elements[1:num_images], start=1):
            image_url = image_element["src"]
            image_page_url = image_element.find_parent("a")["href"]
            self._save_data(image_url, image_page_url)
