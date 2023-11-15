import scrapy


class BibleSpider(scrapy.Spider):
    name = 'bible'

    start_urls = [
        'https://fdocc.ucoz.com/biblia/index.htm',
    ]

    def parse(self, response):
        for p in response.xpath('//blockquote/a/@href'):
            yield response.follow(p, self.parse_content)

    def parse_content(self, response):
        for p in response.xpath('//h3/a/@href'):
            yield response.follow(p, self.parse_content)

        content = ''.join(response.xpath('//blockquote/text()').getall())
        yield { 'content': content }
