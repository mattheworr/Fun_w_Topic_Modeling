import re
import scrapy

class wiredSpider(scrapy.Spider):

    name = 'wired_articles'

    custom_settings = {
        'CONCURRENT_REQUESTS_PER_DOMAIN': 15,
        'CONCURRENT_REQUESTS': 100,
        'CONCURRENT_ITEMS': 200,
        'HTTPCACHE_ENABLED': True
    }

    start_urls = [
        'https://www.wired.com/category/business/page/1/'
    ]

    def parse(self, response):
        '''Parses archives for article links'''
        for href in response.xpath(
            '//*[@id="app-root"]\
            /div/div[4]/div/div[1]/div/div/ul/li/a/@href'
        ).extract():

            yield scrapy.Request(
                url='https://www.wired.com'+href,
                callback=self.parse_page,
                meta={'url': 'https://www.wired.com'+href}
            )

        next_url = response.xpath(
            '//*[@id="app-root"]\
            /div/div[4]/div/div[1]/div/nav/ul/li[2]/a/@href'
        ).extract()[0]

        yield scrapy.Request(
            url='https://www.wired.com'+next_url,
            callback=self.parse
        )

    def parse_page(self, response):
        '''Parses individual article for data'''
        yield {
            'url': response.request.meta['url'],
            'title': ''.join(response.xpath(
                '//div[1]//header//h1//text()'
                ).extract()),
            'date': response.xpath(
                '//div[1]//time//text()').re(r'\w+[.]\w+[.]\w+')[0],
            'author': re.sub('[\t\n]+','',response.xpath(
                '//div[1]//ul/li[1]//span[2]//text()'
                ).extract_first()),
            'body': re.sub(' +',' ',' '.join(response.xpath(
                '//div[1]//article//p//text()'
                ).extract())),
            'tags': ' '.join(response.xpath(
                '//*[@id="app-root"]\
                /div/div[5]/div/div[2]/main/div[2]/ul/li/a//text()'
                ).extract())
        }
