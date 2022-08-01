import re

from transliterate import translit


def clean_city(c):
    c = ''.join(i for i in c if i.isalpha() or i == ' ')
    c = c.strip()
    return c


def point2country(point, bih, hrv, mne, srb):
    if point.within(bih):
        return 'Bosna i Hercegovina'
    elif point.within(hrv):
        return 'Hrvatska'
    elif point.within(mne):
        return 'Crna Gora'
    elif point.within(srb):
        return 'Srbija'
    return ''


def translit_cyrillic(text):
    if not has_cyrillic(text):
        return text
    return translit(text, 'sr', reversed=True)


def has_cyrillic(text):
    return bool(re.search(r'[а-яА-Я]', text))


def is_retweet(text):
    return bool(re.search(r'RT @\w+', text))


def replace_links(text):
    return re.sub(r'(http|www)[\w.,;@?^=%&:\/\"~+#-]+', 'link', text)


def replace_usernames(text):
    return re.sub(r'@\w+', 'korisnik', text)
