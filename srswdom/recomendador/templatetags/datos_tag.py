from django import template
import requests
from bs4 import BeautifulSoup

register = template.Library()
#AS22683 Koninklijke Philips Electronics N.V. (internet presence)
@register.simple_tag
def obtener_datos_asrank(AsId):
	url = 'http://as-rank.caida.org/asns/'+str(AsId)
	valores = {'url':url}
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	table = soup.find('table', attrs={'class': 'asrank-asn-information-table table-condensed'})
	rows = table.findAll('tr')
	asnumber = rows[0].find('td').text
	valores['asnumber']= asnumber
	asname = rows[1].find('td').text
	valores['asname']= asname
	organization = rows[2].find('td').text
	valores['organization']= organization
	asRank = rows[6].find('td').text
	valores['asRank']= asRank
	return valores
