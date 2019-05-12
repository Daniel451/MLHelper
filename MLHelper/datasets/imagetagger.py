import pathlib
import re
import requests
import shutil
from collections import namedtuple
from tqdm import tqdm
from bs4 import BeautifulSoup
from .bitbots import ImagesetCollection

DatasetInfo = namedtuple('DatasetInfo', ['id', 'name', 'team', 'team_id'])


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


class ImageTaggerAPI:
    def __init__(self):
        self._cookies = {}
        self.base_url = "https://imagetagger.bit-bots.de/"

    def assert_logged_in(self):
        # does not actually check whether session cookie is valid
        assert 'sessionid' in self._cookies

    def _api_get(self, path, **kwargs):
        url = self.base_url + path
        return requests.get(url, cookies=self._cookies, **kwargs)

    def login(self, user, password):
        loginpage = requests.get(self.base_url)
        csrftoken = loginpage.cookies['csrftoken']
        cookies = {'csrftoken': csrftoken}
        data = {'username': user,
                'password': password,
                'csrfmiddlewaretoken': csrftoken}
        loggedinpage = requests.post(
                '{}user/login/'.format(self.base_url),
                data=data,
                cookies=cookies,
                allow_redirects=False,
                headers={'referer': self.base_url})
        try:
            session_id = loggedinpage.cookies['sessionid']
            self._cookies['sessionid'] = session_id
        except KeyError:
            raise RuntimeError('Login failed')

    def _parse_pagination(self, soup):
        page_regex = re.compile(r"Page (\d+) of (\d+)")
        pagination = soup.select_one('.pagination')
        current_text = pagination.select_one('.current').text.strip()
        current_match = page_regex.match(current_text)
        current_page = int(current_match.group(1))
        page_count = int(current_match.group(2))
        next_link = pagination.find('a', string='next')
        if not next_link:
            return None
        return next_link["href"], current_page, page_count

    def _parse_datasets_result_page(self, soup):
        team_url_regex = re.compile(r"/users/team/(\d+)/")
        dataset_url_regex = re.compile(r"/images/imageset/(\d+)/")
        results_table = soup.select_one("table.table")
        datasets = []
        for tr in results_table.select("tr"):
            team_link = tr.select_one('a[href^="/users/team/"]')
            team_name = team_link.text if team_link else None
            team_path = team_link["href"] if team_link else None
            team_id = team_url_regex.match(team_path).group(1) if team_link else None
            dataset_link = tr.select_one('a[href^="/images/imageset/"]')
            dataset_name = dataset_link.text
            dataset_path = dataset_link["href"]
            dataset_id = dataset_url_regex.match(dataset_path).group(1)
            datasets.append(DatasetInfo(id=dataset_id, name=dataset_name,
                                        team_id=team_id, team=team_name))
        return datasets

    def get_image_urls(self, imageset_id):
        self.assert_logged_in()
        url_path = 'images/imagelist/{id}'
        r = self._api_get(url_path.format(id=imageset_id))
        image_urls = r.text.strip(' \n,').replace('\n', '').split(',')
        return image_urls

    def get_datasets(self, limit=None):
        self.assert_logged_in()
        url_path = 'images/imageset/explore/'
        datasets = []
        r = self._api_get(url_path)
        soup = BeautifulSoup(r.content, "html.parser")
        datasets += self._parse_datasets_result_page(soup)
        with tqdm() as progress:
            while limit is None or len(datasets) < limit:
                pagination = self._parse_pagination(soup)
                if not pagination:
                    break
                next_link, cur_page, page_count = pagination
                progress.total = page_count
                assert next_link.startswith('?')
                r = self._api_get(url_path + next_link)
                soup = BeautifulSoup(r.content, "html.parser")
                datasets += self._parse_datasets_result_page(soup)
                progress.update(1)
        return datasets[:limit]

    def _get_filename_from_url(self, url):
        url_split = url.split('?')
        filename = None
        if len(url_split) == 2:
            filename = get_valid_filename(url_split[1])
        return filename

    def download_imagesets(self, imageset_collection: ImagesetCollection):
        self.assert_logged_in()
        ids = imageset_collection.to_ids()
        paths = imageset_collection.to_paths()
        for id, path in tqdm(zip(ids, paths),
                             desc="Downloading imagesets",
                             total=len(ids), unit="imagesets"):
            path = pathlib.Path(path)
            path.mkdir(parents=True, exist_ok=True)
            image_urls = self.get_image_urls(id)
            for i, url in tqdm(enumerate(image_urls),
                               desc="Downloading images",
                               total=len(image_urls),
                               unit="images"):
                filename = self._get_filename_from_url(url) or f'{i}.jpg'
                r = self._api_get(url, stream=True)
                r.raw.decode_content = True
                with open(path / filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
