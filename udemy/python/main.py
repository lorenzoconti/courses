import requests
import bs4
from PIL import Image

# Web Scraping

result = requests.get('http://example.com/')
print(result)

soup = bs4.BeautifulSoup(result.text, 'lxml')

title = soup.select('title')[0].getText()

# Images

img = Image.open('images/example.jpg')
print(img.filename)
print(img.format_description)

img.show()

# Cropping

img.crop((0, 0, 100, 100))

pencils = Image.open('images/pencils.jpg')
print(pencils.size)

width = 1950/3
height = 1300

cropped_pencils = pencils.crop((0, 0, width, height))
cropped_pencils.show()

pencils.paste(im=cropped_pencils, box=(int(1950/2), 0))
pencils.show()

pencils_resized = pencils.resize((3000, 500))
print(pencils_resized.size)
pencils_resized.show()

pencils_resized.save('images/pencils_resized.png')
