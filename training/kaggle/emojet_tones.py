"""Emojet human-labelled tone/style data (runs on Kaggle, CPU, internet on).

Collects public datasets where PEOPLE labelled tone, emotion, sarcasm, condescension, manipulation,
aggression, politeness, dialogue acts or negotiation/support strategies, and maps every label onto
the taxonomy in labels/tones.json (65 tones, 8 styles). Plus real WhatsApp group messages with emojis.

Output /kaggle/working/tones/tones.jsonl.gz:
  {"t": text, "r": message replied to or null, "l": "en"|"hi"|"bn", "s": source,
   "tones": {tone: 1.0 | 0.0}, "style": style or null, "e": [emoji ids]}
A tone missing from "tones" is unknown for that row (not trained), 0.0 = labelled absent.

Built by training/build_tone_data.py (fills in TAXONOMY_B64 and SMOKE).
"""
import base64
import collections
import csv
import glob
import gzip
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

SMOKE = False
TAXONOMY_B64 = "H4sIAAAAAAAC/519fW8bSXrnVyF8QGbmYk9mJrubTYJDcEguweSPXHB7yOKwWBgtsiX2mGRzu0lrNMEePJ6xLNt8f5Eo0rY8MyQlkrJsSX6RbNkG8lG0H+AWuC9wVV2/p+qpEuWV7w+5qpvVVc/7W1W3//3SdT+Kg7Bw6W9SX1xOXSqEJV90L/3PsOBfCRevXA+DtJ8qeV+HhTC/kloMo9R/y4f/7Jc+Tf29VwgLQdrLpXLegp+LU17kp5azXilVyvqpfJjxc6mwXCqWS/Hfprxc4MW+GhOviAdX8nFqMQrzyeA4LEdimVwQl+LUx+XYz6RKYSrvFVMZrySeE3e9QiaVCdPxJ59KAL4KUlG4rKYrRkEYxWIesXJc8v0oFZeXlvy4JLASt8NlL8okq5QETklnOQpKYlgQp8pxUFj6NPWrZP34b1LhcsGPPopTC1HgL6Y+/lkqHebzZYmlnE1Mv5ITSPx56vNfJLPFn1xO/UsgcPcLqX8Jo7xXSP1TFJaLqY+Ifv+W0O/XYZSJP0p9/Jd/lVqW3cupn6UyQd4vSMrLSf7VKwo2/PflnHhypSiWCBdTcgrxzP/+5c/FgH+KvHzei3IrCouMX/LTJcGMj3/22Z8Lklz/IvXxF5998Ysrn3925bOff2IQyXuxRFUz8Erql6kFP+tdFzQTnFMIXU59IRAtlPyvS2Vx0/ta3vr8rwUPI0W1OBUU0rlPE+JFguZ/UfRK2TAXLiXslwP+1nA170dLgoFeTEz/9JIQLLWSEK1/v/S7smKOlLNfZ4N0dh6VJXcEvHk/jr0l/+/kFGExYWkyhxfHflQKrvvJVcaP03K2BS/nFdJ+5rKgbiQIdDkV+XFRdARHw3Ih4wm2xnIq/2svX8wlkv5lSvwgiF+SErYiBCFVDIOCeHShXEp9mSr4ApdynMij70vZFoBlfC+TCwQfFgPx4KeXfi+mLAqQXHji8kI+SG5fljgWBT3kzLGfW7wSe+koWAzSQgAdiH4tJNkXapmS/6xIdksJSy2LZxM4MqEkzqJcfzkoZQVcCgJvaUmgewaITJgPCsm62VDQPSdgEdzPiTXeQ5X/FZYV6mK9uBQKRRREWUoFJaVny96KEKO0J1RV3BNythyFQpE4Ia6cA46gSUmIsfgpkDQXYAUFxS1xFxA6wPyjQPVy6suPcrkE95Jik4BBkFIIpi+vhbCJG3HqWiFcFqARV4RKBsWyQNYFoyyt0VI5yJVSQo0yftpPpEviK1UhChPjJZh2hjT/dbFEvBHCIIjy5UfXhUZKPknzKGC5nACU9gofleS4goQ6kRtlgMQzf5cAlw5zwnQKRTwDXToMhUFI7l9O5cLwWpzM7QmGF66IvzkSkwoWk2VL3jVl5eJckPGV4fxSQSDtRj68LHuF1LJQZf964C8LlJd8cStSQJUiryBkUxLDy1lAFcuRL3i0WC7g18up35WDjDDAoeiEDky/8gvK7AYF5UQSQCQPkxUWpVVK/ETBl0Yt461AisXojJALa2n/uqe0KJbmaNnLCe1bupyIfybyll0WCdck8RPOTJAjdy0liFyG6kbBUrYkflkWq/1ermfscLIiOcBfSeoJtyLM2Geffvp5Kk4LYyicU6LxiRVP/ZfUZ8LSzrP/H0l2CRFVtlovIByHRJlsc2lZyFo5VyoLS/wXjv1NbKagdGHlqjB0QViW0P1G3UnMKW7+Vg6TK+eupr24nHDsN7gjx+GmHAZTuFjOXQ2iSMqvn1D5N+wX+Qj7UT4m2qxwk8KPBOmrea8khP9quHh1UYhI8jD//VKicXKI9H3JEDlFVtD8ai5cvgokk+fkzSu4kfo4kMYxHSQW28sIBS4t+4KoklzS0MafyKnFFOYJ/2t6Iie9uZf7JFlLjvaiq+kgEqT1omQpdS/1cVyKymlBbukgElUMi0E6mZmGpz4ueBHpXSYgAyamlqJiOy6pcrEJLITHdvzVopcPcgGcnnApfi4nqKx4dOlLdp36M6EcXm6lJN2pfLIYxgFswqV/RV8M8gtL3hL8BVlKMeIf/EUpYMkQZkAhBNJ+Jj3xq/FHcsD1ck4IoregZvk3fSVXyocwAIkECI++IhmoVit56azwDGLU11nhAUp+RqlfKfLzyVT/Q4Zlf5biQYIyLTJSSKgRe1FaSYy8Sgi1Ip+0qCSNgYogEi4GkQw3ExEL09dAhljw61qiEUJkhftPyVErSnBlmKge/ePW5ndyjGi/V22vrtrhD7huqXb2SLG64JeFmcp9AHjC3GeChBJpIW/EyQyolfQTL6ckQ0+xKIzhtUTtCgnDSmCOo0Yy/Fn4SjyOn4Uoe9KSiWdkfDsH4ZsKoWYVbUW2p/dvqcu2uvz++P8c1RXG6RUF9MUxzgRSoWKyG195GYWnYG6GWCXNZkEOOAvg9zblhy9xvXd6o/rHrdYOGDLUjErAFCa/tPIBQArq+aXEVCackZYt4UWkzKiIqRPiCkcm5egMlCQnPUA7bOP6viInoOzdVNAxNb44iF5aUC6v6CXC+qwWHtKdM1ANO2rZ7Qa4eVe1LUhzB0xvr4G3ZXIhF1a2gqAIjNBlNVTYQqg62UGpg+fC1rzBYTx9AMJt3iPCJZDF12Tc9YFyF4oQEkIXl2Nhw4m/IjIJC9dlSDhHI4Y7NtEIUBJAAlgIZgKaMF5lqWHlD4Et8hdzRksTZyU8m9bqJFyNVLes9fkMqK3HNohKmbW0VTbB3gcK0qL0UsKwf5g9XRJxZT7RgoK/pCEU2cC1OWrwELQCQJVtaTfUrYmtqK0OOOt/qMzFYc7PJ6HtUuSBXmF+QSmsIKWY8bo3n2C9hmXW+o80eJsE5gslhhsKOsEV4Z38sxTTrveMeY/gcmQiJ1Mc4R5LCsgVM1kihJROztWPmeLlg6EGcfQK3N4ABQcadAUrj7AuBq2fFgAAnsR9E3D+1yJkUlpsI1EuLvheaY7WgL2dEYCqqLZ2R1EaFqf1hGBNi8BXRSkXAzUuF4thRL+I6C+Oy0TqoCC0O/JMKCLDEikD8wjbmnIIjbJsgboPGd2V2uS8lbPafa4AZH1fRRoiCE+ss/S/WRFYgp5CAdPZwL8OW1QSmACN5WyQj+dbcuVHRPsA7SqgndpxyiYsebacD6OzanUe0DpfEEJLXsXLJ1WvObAgZBj+ZMPSgmHsHWtfmMDyVfiB5ItgtLNesbiioqOEiPOsde+u7XhHh7j+NmFj74ZWHyGJCTi58PrFpc5bXPRVFqvEPwpF8lYiHyw1OGGcF+Vd0CzNfQqgeqBUF9fwv611hAXFogj9ApPn/2kAhQks+aBXKStixPlBFFm3u7Z2tmAPR8eu0HuZfBBdnE4ity8n0i0SNJkMwXSI4HO+DDUBUH3VpoQG7InWygQcWdOb67kuJEdhkUj0lWC+kzudNWVwYpU1O3a760qUADYBbjEKhCjkVi4I2rKfk2pGXqIgK+vMdiEVN6G+umsqPecrQfOe7SGUUpzerxiQEYCKZReDjI/yyQWsryxFJiZXcXYhzGWUE/OKMndL/MZZuGrg5tR2DkNYsWbT9gqLuSAqrVwYpEzZRFBZoYKapO8L0O/YqjiDEWvdsw1oQSTIXm7pwvImclqRzcVpSdIkqhFWIi/+UdxbDlT6cwasEWxl+whgIKLrPgXLjBnRsbDgnax3+SpwsMFjeb1lxqIoXPIUaCIGkRnyN5QZ58uqLRdlJJbYM1kMkxXLTLhceE9G5uTGFDBToNy6peDNUN3hYrAulUVeqKTsK5E1Bosr873Q0MnU3fxwVxHtHpnW+XHcOUAshBD0NARdJLGsJCJL0cIzROB2kk+IgL0w10M1bMBG+wCwaSPQGgLSdLos7EEYrVwQ1JyX13GQysbCCEYu8oVh9tLZubLXewTAYOCapBpDm6KU6DiF+j8tdAuRRwRbENF8sKQJJuxa2tdePoqzvFSVOI7wepjWyUaxvFTwdPZWyoroTzjf+WIBYRzugqo/Arm7NnLCFCETX7oooRdlEUO620QgvEIhXFHdIIqCkjc/PHnk0PQHGywtvRPoShDLIGBeiHIOVGmxNoVsX5UzSzA6ZyFp2Ss2YZuHY1scZxDDhUAWli4GgxALsSql2sVARSTnguCqbqvjgPCIDEfe+wAjtxDG8QpiECE6ELViOc6e8Qen9ac2KEr2ZdpnswWQROXMBYVeMNAuk6dDId++UzCX0GSWVlBdXfKis8Ra79qgkEBr63rDLnh55VI2TOTwrIKaMMKSnDBP9E2InQ5IW5Ny3bLItrJzPGnbrhCS1WidgHk/WAm0ri7/aXiE2C/6KtdEhJYL4qz/J4uWplhZs0GhxJgx5EJwhHKnQPGuLLI22j+ZG1ETKDoEq/IQR/tsFiVdjDWFUgQgIl+YXZImEWaEQpxI1cIoWjkfNioYUDVm8zauH9olmJI0GRcWm6wvPAxiibi8kCkrIyjjc4SEwhOWPLX5WC7IkPH86l/vFvfWJhzTXvyFArEcLZ0NVs8TIylESqi9nAgLz0qQKfxA47U9WEVYWE/2AU5vnOiEKPYyzvJsX+RM5XE5wT7J+hei8JqfXOZ9eewgK8QrbRetJB/D5fk2E46i99jmJdlMzeOH5M++nlNOOx9UoeUie8mo8l5ERQkTzijxiOclHWRCt2wWEkQ652yDgOWoGAUyM7wobHE2TF+Dt81736Ant3bPsQu9PSx5AClCFtk7tEu3vX3ubeUxjg+AKueXUsRgMYGqY9HOTawKPfMDQac4SpVQqowyN6hztPKH0KtY/uabnIIkJ7zTHAi6Tkn7udpDmXGl0yV4Ec+jBrogNzvjDwHFW76W7DdJvotMvCgIMwecQ51zWcZAG4GxlizlS/QezQfs6+SCYtErzKtYTvkWjdn602nNS3vPpuhnVOnn/2/HphCUioHajzwLSluVFh7oPL3dd3ZuHuhYXSlTKfwgSMoFa6tWnqpZKqhjO4uytHVu1rLxjrt1qo8b6mz3kC8H2SD3oVsLQnC8a8oyejKczwnLPScSgp50DzV5KPiZUXj9IhHkykyPqDR18hl4ZzzHuXFskqfACAbCd4k8UARI8yk0VB4ejp/iNV1evmUnGsItZT4gq/Oue5HZsBLxfTo5nTQ35QHKDYQeY6hTCwZ6iLC6t0YkiYKLZ3FsEz0KMoEA40JFYb1hfmynG5RMsgL+BQt6YSEOc2ZvZVFuBuDKy+Xn6xXC+dnNMzsmCriKvcmIXbcL7p14wkHGJe3nF/2cugiT2+dEZVZZGLSZvbBjSCqHSZSTfQ2Vz1+0LJYW4SJ+yJZjHCoICqmwkArm2cGZbffISQ4RcWze1ZUzFWHkRHrwIaVYWTtMNFvu2AXX53PqwCUO8YiIU6FaOTYz7T2fixaFhVLFedrxol0ibRXDc3Y83Ppl/wbfM9IVYqoHhwux/7vyHEc1P2CNV9JhMau3FRa95QIEezGXnO6YC5NOQG7bwkPhF1XYWw2dBCUm/+JBrL/oU7FDVtDnG2cKZnRw4xzLeOQUvw5wPENojGsJ2fmhsyGX4NwSbU2y00SlIJobaiGk6j1zam42dL2pXXDwcmG4eGGo7DM7MeVEScXwXH9K/lOn9eRHTf74W3UgNZ0NvOvSHXkFFyQcoTpjIXOZ3ErKOsqaHK3J+vPNY2/N1nxyILVE8083t7QyChuhoPKXBI3kCt7ZsxjngLUUySJDqGogyT5CkJc7uXODMtoR0DsHt2ztp/0i8qppT2SuySGiC8EiFcooFvKbb/wopJNsc4hExhkMnI3swKg1ZDL0+zNPU8bxnb1NqbYnzd4phb56T/WmXV6g4JT0aTZztjPISN92dhypHDGxtz16Ku3tvTLnMxBx9ja1pqprbIz0Bna4rMPnB3aUQT6X3MfsRyeon9k5CAWWw4ZdjdYK07QVh+h/NuTTWzxOpZMSU0f5h5hoRtb7pvj5dLXD9sqSG12zQMVO4Mh8DIe2tZkRsaHbQ9qaoPQLJBkiXR1ug+TP0QLV0TNK0+ana8huh4/sre/Rka1Dw7azP9N18k8k8JvYMj82CFM+/cTOqylxm01tn0MBTY8SNyDWI56SVBIP3eIGmel1tD85zqVtZ8y9mWPenSq73uHY1RGwMrLveBpx2jPHgFoTO3xtUvhKXvU12jdo32I8QUzlmCPnLASe66E02cPzm2QfAFHvrZ0Sb96xg7AW7Y/3dDREJwhUS2eDQKnWQ/vEQUtFLA9+Mhh3TOSViJcKctiNybEbm03Iaf1gx9Ut2JAWbMgEgtsimwGJpIxuQslk1d6Epd1/2iWZ0TXkhjSgBTkjleYZIlje/JbQ2NjRP/LuE/PIY9ui6910YLXe0Dux+GUdVJ9B/mbQjxnpBxkE0Gt2xA/Zipmprbv79HQGYUY5HJ1OAt+HoHBzza7CN1ftbLR5260szJ47B37BkxklI6+stJb2zYZg8fC+HV7SgQ0K05tkxg/srX5kP/erhtzk8g7txHkMFRhDJcYAeEwrwCWOSRXhGrfJNVJtAC52DOKPn9n5TSIZ6NJeBjzGDHLVxJLbdPST7sOubcM3NcH/Jh0dpXO3HelBNm/qhXDDHCFp7iSyOX5qX+7bl4f2JUjbnFBrzbHdtKecWFPSr/v2r4f2r4f2rwf2r7T+vovMvosurFCTSA9WbtJpn6rzAG6wIzZUNqi5I2vuyFvUuiNvuSNXqXVHrjojW99S64zEDTYnHR+55855zxm5rVuXWHV3ztvUunPedkZSoXQ4dkbiBhtJIclLd+RLF07J7NN+1xZV68bEuQH5qLUtcdWXE+sSg+sza7C+nFiXAGbdBWbdBWbdBaby1ppfX06sSwyu2opWtdWgaitJZ9sarC8n1iWh+dhG87GN5mNrMO1fYrC+nFiXBMauDcauDcauPfNre+bX9syvbT4ObT4ObT4O7cE79uAde/AO5+P9NYeP/MbEuYH5qeSC+fXlxLqkwS17cMseTJd71Dq6gBtGF7rP7a4z3txjpoNKjjdd03HTGTmCSjZ164LjquaMPCeZVQorSF4P3RkOXUtNFp0ykufUumbhufskuZWn7hpPXVODkTNKXaniYbueqi3sVVu69a9oa4T6C1v6MPf4iFoHOtww0I2PqXVHHruGcJNa12BvuiNp06bvjuy7IwfUuiMH7sj71Loj77sjH1DrjnzgjnxIrTvyoTtyi1p35JY7kuJA2nygQ+Guk22dcbK3qXVHum6u/4JaZyRu3HBuPPxh3oj5v9w8+8t2lVoXeTdK2a5R646snRnpLlI7DzA2mfsL4rAnzqUzjMLFJ86lO2xiD5vMH0arnDiX5yx64lyes+iJc3nOom+cy3MWfeNcnrPoG+fSGVb/jlqHm7hxw7lhPXoeN9lk83GcwFy2sHoXlYTuMwM/seiJa3ufuGK2Tq0rkOvuyA1q3ZEbrj0HQHWVUg5eGtzg6+omf026LvkqZ91k/Xtq3cHfuyr9E7Wu8v/kznmXWnfOu9bI08Gx1bNGs1ts5nt2153/3tlH+s+odcF+5o58Tq070vXBwyNqXW/ternhCbXuyBN35Btq3ZFv3JFU8nrrjnzrjjym1h15xsf2qHWlsOeMHCDSG1Qs7Rk+SBqdkFN8TekEnQa2okxT4mrdS5omD2Lf8/M5T9M5nub8YefNft60Or/fwaVqxnPvbs+/+75Hxu+db+60O3Mf2Xnf2O33PTJ+73z2tJO5VJjMpcLkfVSYvI8KkwtQwQF3PhLOM/PJMB9FB5b5EDrPzMfRnn/DVH2blMyN7BMSzamlVtS05t+d+6N7ad1tworPEB42YPQa+H1MZ1gAcgMhXwOJyniPrpXvhk9qQFUauB7DBYwREzeQczTWqJWqtosofIwyTwP+tYEko0FAUhmXSqW0X4rrMWUfKG2MYZvGsDoNAhpEbsDFN/BcA5lfA167AZ/cgE1u0HowLA3UWBtgVgNWboxMZQxiNmBVxyD6mFIc4N2AlWwQ/rCtDdjtBkKSBiL0xjuTJdGSqHSOachrahNb9r1+oAEj14BHGkPwxpDYMX4fE8zYWmhQNZhohgUboHUDuDUg0A3M28C8DQhS966BHewdg71j7DKMUUMfY6opZG8M4RojvRsjJZsSe8ZcmKbv2F1zZKQBn92AZDZo0Zq9995AZt4gYEjwSTaQ4zdQcGzQwWlSHOIpMrcGquKNH+zDYQ3ISgNITQHflPgDWRgjVxxD5saQwTHJHsk64JnhukEyBmI1oIlTwo/mhaxPgVfXhJXdI8MyqvkDqykkcwq9Hj+y32et0Ot00LwpNKxuCjkVCPmIzvLAdFQgwRUQooJHp3QfS05hFCpAqAIEKnQflo1KjJV3aiempyGoYuUq1K8KJKsQ9SkgnAKyKszFFDNWoXNVsLQKkalCxKr3qE0kcQtiXAWHqpCYESSmCkmsEnigZRUcr8KqVKF5M3p3GJyt0uldjBtB8kf0PNRtRCfXIdkVqFHlhSlw4ZERfW0HS4zoiDmUexsU2cbSIwjbDEakQiUoCPlsQJRUFIERm+GxKmRyRJjR9h4wHsFGjeAnRlRHASO2iaF4rgrZHoHyI8xTBQWqkOYqUQKcGEG3KpDeCtabEeWoIkUUBXpUoB4B/SptR2L+EWzlCHBVYaq2IUDbEMAR6FKFqFdpPqxbpfWInEQ/kgRoYxU2rop1qzCsVZIM4FeFbRqBHiM60ID7VbC7ClteBR+qJNGgV40UiHbCoFgjeo7wghhVSfIJDtCZio6jgVJZOi9BbIVmV+EfqqSPsBRVGJ8q9LdK+gsLMiIyU2GUxIQq6tDbbdrYvGMfERsRWd8YU0Ln3KAQHTrYBo7VX9nHqCrQ9QoZQzrtbc4DbdDWCIhU7+jM+KnJxkjRAPIGUKybTaG62ZKp3zddsyFdf2i65sRZfWC6GyYdJOMLCZiCFQOwpm6sa71vupCnOuSpDgLWIZd1EL4OuaxD3upYpQ45q4OqtCtVhxzR9k0d8kMvE7VgvjYgTxtvFP0wTReGeAD56qq32QYT4wHVR5AG6nDsgJgIsa5QnWrTODXwv0LH5mAhKrBQFfWa46Y5/1ADbQbQ3RpoVAONWrANNQDdh/D2sXgfK/axYh8r9rFiH8/3IcR9CHEf6/VhQ/uQoD6I0oek9iGpfUhoH66BNnH6sKl92MY+iNmHRPShXH1Iah8aMYCk9mHL+rBlfdiyes2IjzmzMKCzCPBZVL8k1vRhcQfQxwFMwADj+jSPycQG5ljN4EetYph/QNSBJPbpADIkdEDUgIUbbKqv7YAoA3Xah6AeUP0c8j441OuZIuNgZApPP6ozG2YzjsnmAEowgBJMacOLSAEp7cOo9yFo9CUgZkVo924AwzkwtN9OXNppRVniOq767J290/o+YVE/MJpjZu+Cs13KBCBJXXC6Cxp2IZFdSFYXtO+CmF0g0oWkdYFQl9QXiHUheV1oUBcS2IW16UISu/RaCySyC952IZHdsX6TRbWAqwJ4KvSNLYoesG4F61V69lfAKpi3AnpUgH+FStH80yYVEKMCIkzV19keN+yPJVVAg4p6kaL3rRaoW6oxolMxcl4xIlYZm675VljF2KeKkbjK1HTZiz67pnuDf7hFtOoTWj1d01UHN08HWmL6eFn5ez3iO42B+nZKz6Sw03v2WwAtCFQFJq0Gk0ibkTWYwhqFhzvW2wK1Nf6FLNHSNFimRqexYQFrZLHBgxrkYhtg1IB9DYpXM26zZvxqDWpbY24ZcNYpiqYzxABwhE94Ke/VR1wwAlh1LFsHWHU8VSe0aDwBgCC1TqsDjTqlDXTSG2Jeh1EYUfQD8owgYcokDEyuWKODxohua7ClNVwP6FUE+kIZvPYUiExBvw5RCjFHh3JJYi+uu+a8bA2Ov0bb6wgctim5BoWnuD9dVe5Yv428OdK9se79pHuGjd+hvUHAPONH9/+4tX5igKKzF/ACU/pcC7zBlL6QQXbO5GEjkG5ERQE6BUvfbqRThrTdBNJvUyQLTdmmsiRMUxOmaQQnPAGJJ6DbhA4y0ClHKlJQWR8mrEmHEWGtBsZT1yBOEzojAUPWpByDxBJS0YQla4I6tB3XBHWme/Yni5qYvwZW6mNNNH5dyWXTvNJJh47pqxd06BWK1IHadaBAHShQB4LfxsxtELkNue2AGR0wo0ZHJcDvGvhcM2FizcTGNRNn04meGgxKG9ypQWFq4P4UAE3BzRqku4Zgp4bnppR2QqGm4PoUiE6B6BSItiEFbRXu9kwU0Abz2mBeG8zqwBZMoaodOh0NCNaNo1knl3LKnMv6vuma2KEFqW9BRFpgXJsOGUEUapDumklg2iYlqyEuaoNq7Zf2l57a9OWnE/tbrB2IQ5cdTKbPtVJeQpYInG9TC4K3oY5tMKgNwrc37ZeL25DZNmS5DQfWhlq1Ialt6MIGQxRq0KHEy5Ttppv266dTepUHmtiiQ+GQyxYI06KaHR2zI/Kr4+f375jFSUBh2drQwTYEuA3GtEGvNsSjjRymTayBOWmDnhs6Hr5f172WERGz+7tuqsnrZtN73Zgfel97WrE/SdKGLrQRLW+Yrc42fcUE7njDrLzRNqNu60/2qvaO/Wnf9j3+hVPRVu1vo6zfNlNRq2K7+waUje9M18Q/G8aWdSAxHUhQh+4jCOhAPDZMmj8l+abUXb3zPtCU7ht8B+aFjA0dnfbNmzktmMApZcCUZplTFVPIOJ3DnMJldCZqMmMMN+7rFUxMRKdhOuo9qUFb7/WOzA3MTOeuqOoEJzalbQJMzqoYtMlGp1shj3RYs0NySa9XgMcT0hDwegK0JxCuCUzNhHJbrDOgGjsF8zCjmvaia2R7SjklzOCUEjeKn8ic0zYJbfhBv7fp5RBAtw2otsktUJBAHwwiDlGwQId9wIIBZfrqg259k9lNx+qOseLbgHG6a7+kN6U1qIKD6z40v69SyU0M7wP0PuWyRFBK42GaBtDiAeRwAIOJ7Id9Xrh/pMSGqhTgfB+2rg+O94FzH66wT04DNrOjPk7b04WCng4Od1ept0qSZk4Zna729Ljb+t6mvndL31vXva7u6crf6kT3KBc7HT7TPbO50gFBOlSPhE/okFaSNkJHOhTAKVPUN0FJ9462xCZPPdK9J/rXB/qejpx7GsjenpmQwi0K+58o1quPfW6iQKa+9YmPkONLweqthVP1SsIpXmjZVBBvNlXTUg1sGy2FwKFDmYXKS9df6MKFMkr1x/qGxmtDn1ZS1cfTukGlc6LuUPSjEjRVlTytayLV9TGn+ivdM6pfg5fs0Acr4H/a8M8DigYgq+1nCkcNmDkz1d/W7NBy9LCrGhUPP4Qp2oLCvaVhNyhhOf1Wc1x9iOb0Ad5Q67IPNv3hhjEDLShLS8WN+oWE083XGjBTee5QlEkiqCB53GWfEjvtNTUiWvgfKCl4oNj/UD328J2C5o0eb2IIrRgPSOX+g/TnP4iJp9//aHa9ErP8n6Sn+S6xkf/ZdD8z3c9N9wvT/UvT/Znp/tx0f2G6f2W6vzTdvzZdQRW4iA5F2hSCIgnp0Dg4gVuGHauQ+lvGLK/CpazClK+22f8qoe4oHrdNDLKKYHYVEeatt+YnCNEtcx6Bvtu1CkOzCkOzBpVbMxOvmVB1DSK+BltwB+HLGuz/Gs0Cl34HeKzBEa/B5a0lqvGHu2R4/3DXhBdr8CFrZMLxzBYouwXKbhFlMW4LFN5Sn5lRHuOUXPoWnOUWHNkWoira5N2CUm/RePW4OqRwuqv+V4oNrXMbWlPUts8ptirWtQVdNxW3DqjWgXPqgBodUK0DjDvwmi2SIfCoA0PSAVXrVCmmVBqhax2Uqpu3OpOuenNnzb5jbI98UN5gb/XeHou/HdafsP6U9Wesv8v6e6y/z/oHrP+M9V+w/kvWP2L9Y9Z/xfonrP+G9RGb3N6W16y/zfoT1p+y/oz1d1n/Mevvsf4T1t9n/QPWP2T9Z6z/gvVfsv4R6x+z/mvWP2H9t6xPuO8w3HcYT3cY7jsM3x2G7w7Dd4fhK/tPWX+f9Q9Y/5D1n7H+c9Z/wfovWf8V679m/RPWf8P6b1mfcJ8wnk4YXhPGrwnDZcLgnzCYJ2zOKaPnlNFzytaasrWmjIZThuOU8XfK+DtluM8YzWcM5hmDecZgnjGYZ2ytXQbzLpP/XSYDuwz+XSYPuwyXXYbLLoNtl8nALoNnl8nALuP7LuP7rgPnEesfs/4r1j9hfeL7Y0aTxwyGxwyGx2ytx2z+x2z+PcbTPUafPUafPYbvHltrj621x3ixx/DdYzDsMXz3GDxP2FpP2PxP2JxPGD2fsvFPGb+eMn49Zfx6yuZ8ymB+6sz5kvVPWP8t65Ne7DMZ22cyts/ouc9g2Gf82mdr7TOa7DOa7DMe7TObsM/gOWAwHLB1DxgfDxitDpicHzC6HTC6HTA4DxjfDxgNDxgNDxiPDhg9D5gMHDB8Dxi+BwzfA4bvAcP3gPHigNnAA4cOxJdDRpNDRpNDRodDRodDRodDxq9Dhvshw/GQ4XjI8Dpk8B8yeJ4xuj1nsD1n8Dxn8Dxn8DxnfHnO+PKcwfbcmf+Q9V+y/hHrH7P+CesTPV8wOF8yOF8yOrxkc75kuL9kcx6xeY6YjhwxvhwxWT1iax0xOhwxOhwxHh0xH3HE6HPE6HPE6HPE6HPEcDlitDpieB0xWh0xmTxicnjE6HbE+H7McD9m+B4zfI8Z348ZvscM32OG4zHD8ZjheMxwPGY4HjMcjxmOxwyvY4bXMePdMcPlFcPlFYPzFVv3FVv3FaPhK0afV2zO12zO14w+r5kMvGZrvWZ8f83Wes1k74TR84TB8IbR7S2b/y2jwzsGzzuG1ztDk/qz//vo223x96P4G4m/PVy/m/P7ofj7Sfw9O+f3l+JvF+PeXfrt7/8fKZrKnD99AAA="
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "emoji==2.14.1"], check=False)
import emoji  # noqa: E402
from datasets import load_dataset  # noqa: E402

T0 = time.time()
random.seed(11)
csv.field_size_limit(10_000_000)
OUT = "/kaggle/working/tones"
os.makedirs(OUT, exist_ok=True)
TAX = json.loads(gzip.decompress(base64.b64decode(TAXONOMY_B64)).decode())
TONES, STYLES, EMOJIS = set(TAX["tones"]), set(TAX["styles"]["options"]), TAX["emojis"]
SKIN = {chr(c) for c in range(0x1F3FB, 0x1F400)}
E_INDEX = {}
for i, e in enumerate(EMOJIS):
    E_INDEX.setdefault("".join(ch for ch in e if ch not in SKIN and ch != "️"), i)
CAP = 300 if SMOKE else None
rows, stats = [], collections.Counter()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def clean(t):
    t = re.sub(r"(https?://\S+|www\.\S+)", " ", str(t or ""))
    t = re.sub(r"(^|\s)@\w+", " ", t)
    return re.sub(r"\s+", " ", emoji.replace_emoji(t, " ")).strip()


def emoji_labels(t):
    out = []
    for m in emoji.emoji_list(t or ""):
        i = E_INDEX.get("".join(ch for ch in m["emoji"] if ch not in SKIN and ch != "️"))
        if i is not None and i not in out:
            out.append(i)
    return out[:6]


def add(src, text, lang, tones=None, style=None, reply=None, e=None):
    text = clean(text)
    if not 2 <= len(text) <= 1500:
        return
    tones = {k: float(v) for k, v in (tones or {}).items()}
    bad = [k for k in tones if k not in TONES] + ([style] if style and style not in STYLES else [])
    assert not bad, (src, bad)                        # a mapping typo must fail loudly, not train garbage
    if not tones and not style and not e:
        return
    rows.append({"t": text, "r": clean(reply)[:500] or None if reply else None, "l": lang, "s": src,
                 "tones": tones, "style": style, "e": e or []})
    stats[src] += 1


def capped(it, n):
    for k, x in enumerate(it):
        if (CAP and k >= CAP) or (n and k >= n):
            return
        yield x


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "emojet"})
    return urllib.request.urlopen(req, timeout=300).read()


def run(name, fn):
    t, n0 = time.time(), len(rows)
    try:
        fn()
    except Exception as exc:
        log(f"!! {name} failed: {exc!r}")
    log(f"{name}: +{len(rows) - n0:,} rows ({time.time() - t:.0f} s)")


def hf(repo, config=None, split="train"):
    return load_dataset(repo, config, split=split)


def on(*names):
    return {n: 1.0 for n in names}


GH = "https://raw.githubusercontent.com"


# ---------------------------------------------------------------- English
def isarcasm():
    data = fetch(f"{GH}/iabufarha/iSarcasmEval/main/train/train.En.csv").decode("utf-8", "replace")
    for r in capped(csv.DictReader(io.StringIO(data)), None):
        if r.get("sarcastic") == "1":
            t = on("sarcastic")
            if r.get("satire") == "1" or r.get("understatement") == "1":
                t["facetious"] = 1.0
            add("isarcasm", r["tweet"], "en", t)
        elif r.get("sarcastic") == "0":
            add("isarcasm", r["tweet"], "en", {"sarcastic": 0.0})


def sarc():
    ds = hf("marcbishara/sarcasm-on-reddit", split="sft_train").shuffle(seed=1)
    for r in capped(ds, 60_000):
        add("sarc", r["comment"], "en", {"sarcastic": float(r["label"] == 1)}, reply=r["parent_comment"])


def talkdown():
    tar = tarfile.open(fileobj=io.BytesIO(fetch("https://nlp.stanford.edu/~zijwang/talkdown/talkdown.tar.gz")))
    for name in ("balanced_train.jsonl", "balanced_dev.jsonl"):
        for line in capped(tar.extractfile(name).read().decode("utf-8", "replace").splitlines(), None):
            r = json.loads(line)
            yes = str(r.get("label")) == "True"
            add("talkdown", r["quotedpost"], "en", {"condescending": float(yes)})


MM_TECH = {"Accusation": ["accusatory"], "Denial": ["defiant", "defensive"], "Evasion": [], "Feigning Innocence": ["defensive"],
           "Rationalization": ["defensive"], "Playing Victim Role": ["defensive"], "Playing Servant Role": ["obsequious"],
           "Shaming or Belittlement": ["derisive", "condescending"], "Intimidation": ["aggressive", "authoritative"],
           "Brandishing Anger": ["angry", "aggressive"], "Persuasion or Seduction": ["flirty"]}


def mentalmanip():
    for r in capped(hf("audreyeleven/MentalManip", "mentalmanip_detailed"), None):
        votes = [r.get(f"manipulative_{k}") for k in (1, 2, 3)]
        manip = sum(int(v or 0) for v in votes) >= 2
        techs = collections.Counter(t.strip() for k in (1, 2, 3) for t in str(r.get(f"technique_{k}") or "").split(",") if t.strip())
        tones, style = {}, ("manipulative" if manip else None)
        for tech, n in techs.items():
            if n >= 2:
                for tone in MM_TECH.get(tech, []):
                    tones[tone] = 1.0
                if tech == "Evasion":
                    style = "avoidant"
        if manip and sum(techs.values()) and "machiavellian" not in tones and techs.get("Persuasion or Seduction", 0) + techs.get("Rationalization", 0) >= 2:
            tones["machiavellian"] = 1.0
        add("mentalmanip", r["dialogue"], "en", tones, style)


CASINO = {"promote-coordination": ("collaborative", []), "empathy-coordination": ("collaborative", ["empathetic"]),
          "showing-empathy": (None, ["empathetic"]), "small-talk": (None, ["friendly"]), "self-need": ("transactional", []),
          "other-need": ("transactional", []), "no-need": ("transactional", []), "elicit-pref": (None, ["curious"]),
          "vouch-fair": ("assertive", []), "uv-part": ("manipulative", [])}


def casino():
    for r in capped(hf("kchawla123/casino"), None):
        prev = None
        for text, labs in r.get("annotations") or []:
            styles, tones = [], {}
            for lab in str(labs).split(","):
                st, ts = CASINO.get(lab.strip(), (None, []))
                if st: styles.append(st)
                for t in ts: tones[t] = 1.0
            style = collections.Counter(styles).most_common(1)[0][0] if styles else None
            add("casino", text, "en", tones, style, reply=prev)
            prev = text


ESC_STRAT = {"Affirmation and Reassurance": ["reassuring"], "Reflection of feelings": ["empathetic"],
             "Self-disclosure": ["sincere"], "Providing Suggestions": ["encouraging"], "Question": ["curious"]}
ESC_EMO = {"anxiety": ["anxious"], "depression": ["sad", "resigned"], "sadness": ["sad"], "anger": ["angry"],
           "fear": ["anxious"], "shame": ["embarrassed"], "disgust": ["disapproving"], "nervousness": ["anxious"],
           "pain": ["sad"], "guilt": ["apologetic"], "jealousy": ["bitter"]}


def esconv():
    for r in capped(hf("thu-coai/esconv"), None):
        d = json.loads(r["text"])
        prev = None
        for turn in d["dialog"]:
            if turn["speaker"] == "sys" and turn.get("strategy") in ESC_STRAT:
                add("esconv", turn["text"], "en", on(*ESC_STRAT[turn["strategy"]]), reply=prev)
            elif turn["speaker"] == "usr" and len(turn["text"]) > 25:
                add("esconv", turn["text"], "en", on(*ESC_EMO.get(d.get("emotion_type"), [])))
            prev = turn["text"]


ED = {"nostalgic": ["nostalgic"], "sentimental": ["nostalgic", "loving"], "caring": ["empathetic", "loving"],
      "grateful": ["appreciative"], "apprehensive": ["anxious"], "anxious": ["anxious"], "terrified": ["anxious"],
      "afraid": ["anxious"], "guilty": ["apologetic"], "ashamed": ["embarrassed"], "embarrassed": ["embarrassed"],
      "disappointed": ["disappointed"], "lonely": ["sad"], "hopeful": ["optimistic"], "proud": ["confident", "joyful"],
      "jealous": ["bitter"], "devastated": ["sad", "resigned"], "content": ["joyful"], "impressed": ["admiring"],
      "excited": ["enthusiastic"], "furious": ["angry"], "angry": ["angry"], "annoyed": ["angry"],
      "disgusted": ["disapproving"], "surprised": ["surprised"], "joyful": ["joyful"], "trusting": ["sincere"],
      "faithful": ["sincere"], "confident": ["confident"], "prepared": ["confident"], "anticipating": ["enthusiastic"],
      "sad": ["sad"]}


def empathetic_dialogues():
    for r in capped(hf("Estwld/empathetic_dialogues_llm"), None):
        conv = r["conversations"]
        if conv and conv[0]["role"] == "user":
            add("empathetic_dialogues", conv[0]["content"], "en", on(*ED.get(r["emotion"], [])))
        if len(conv) > 1 and conv[1]["role"] == "assistant":
            add("empathetic_dialogues", conv[1]["content"], "en", {"empathetic": 1.0}, reply=conv[0]["content"])


SE18 = {"anger": "angry", "disgust": "disapproving", "fear": "anxious", "joy": "joyful", "love": "loving",
        "optimism": "optimistic", "pessimism": "pessimistic", "sadness": "sad", "surprise": "surprised", "trust": "sincere",
        "anticipation": "enthusiastic"}


def semeval18():
    for r in capped(hf("akkasi/sem_eval_2018"), None):
        idx = {int(k): v for k, v in eval(r["idx2label"]).items()}   # dataset stores the mapping as a Python dict string
        add("semeval18", r["text"], "en", {SE18[idx[i]]: float(v) for i, v in enumerate(r["labels"]) if idx[i] in SE18})


def hate_measure():
    ds = hf("ucberkeley-dlab/measuring-hate-speech")
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    text = {}
    for r in ds:
        c = r["comment_id"]; text[c] = r["text"]
        for k in ("respect", "insult", "humiliate", "status", "sentiment"):
            agg[c][k].append(r[k] or 0)
    for c in capped(list(agg), 30_000):
        m = {k: sum(v) / len(v) for k, v in agg[c].items()}            # 0..4 Likert, mean over raters
        tones = {"derisive": float(m["humiliate"] >= 2.5), "condescending": float(m["status"] >= 2.5),
                 "rude": float(m["insult"] >= 2.5)}
        if m["respect"] <= 0.5 and m["insult"] <= 0.5:
            tones["respectful"] = 1.0
        add("measuring_hate", text[c], "en", tones)


def politeness():
    base = "https://huggingface.co/datasets/Cleanlab/stanford-politeness/resolve/main/fine-tuning/"
    for fname in ("train.csv", "test.csv"):           # the repo mixes CSV layouts: read these two files directly
        for r in capped(csv.DictReader(io.StringIO(fetch(base + fname).decode("utf-8", "replace"))), None):
            q = r["prompt"].replace("Question:", "").strip()
            lab = r["completion"].strip()
            if lab == "polite":
                add("politeness", q, "en", {"respectful": 1.0, "rude": 0.0})
            elif lab == "impolite":
                add("politeness", q, "en", {"rude": 1.0, "respectful": 0.0})


SWDA = {"fa": {"apologetic": 1.0}, "ft": {"appreciative": 1.0}, "h": {"tentative": 1.0}, "ad": {"demanding": 1.0},
        "ba": {"appreciative": 1.0, "admiring": 1.0}, "qw": {"curious": 1.0}, "qy": {"curious": 1.0}, "qo": {"curious": 1.0}}


def swda():
    ds = hf("nico8771/swda_processed_v2")
    names = ds.features["act_tag"].names
    for r in capped(ds, None):
        tag = names[r["act_tag"]]
        if tag in SWDA and len(r["text"]) > 12:
            add("swda", r["text"], "en", SWDA[tag])


def dreaddit():
    for r in capped(hf("andreagasparini/dreaddit"), None):
        add("dreaddit", r["text"], "en", {"anxious": float(r["label"] == 1)})


DD_EMO = {"anger": "angry", "disgust": "disapproving", "fear": "anxious", "happiness": "joyful", "sadness": "sad",
          "surprise": "surprised"}


def dailydialog():
    z = zipfile.ZipFile(io.BytesIO(fetch("https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip")))
    dialogs = json.loads(z.read("data/dialogues.json"))
    neutral = 0
    for d in capped(dialogs, None):
        prev = None
        for turn in d["turns"]:
            emo = turn.get("emotion")
            acts = {a["intent"] for a in turn.get("dialogue_acts", {}).get("binary", [])}
            tones = {DD_EMO[emo]: 1.0} if emo in DD_EMO else {}
            if "question" in acts:
                tones["curious"] = 1.0
            if emo == "no emotion" and not tones and neutral < 15_000:
                tones["neutral"] = 1.0; neutral += 1
            add("dailydialog", turn["utterance"], "en", tones, reply=prev)
            prev = turn["utterance"]


# ---------------------------------------------------------------- Hindi / Hinglish / Bengali
def trac2():
    for lang_dir, lang in (("IBEN", "bn"), ("HIN", "hi"), ("ENG", "en")):
        for split in ("train", "dev"):
            data = fetch(f"{GH}/socialmediaie/TRAC2020/master/{lang_dir}/Sub-task%20A/{split}.tsv").decode("utf-8", "replace")
            for cols in capped(csv.reader(io.StringIO(data), delimiter="\t"), None):
                lab = next((c for c in cols if c in ("NAG", "CAG", "OAG")), None)
                if not lab:
                    continue
                text = max(cols, key=len)
                if lab == "OAG":
                    add("trac2", text, lang, {"aggressive": 1.0}, "aggressive")
                elif lab == "CAG":
                    add("trac2", text, lang, {}, "passive-aggressive")          # covert aggression
                else:
                    add("trac2", text, lang, {"aggressive": 0.0})


def hindi_sarcasm():
    base = f"{GH}/pragyakatyayan/Tweets_Dataset_for_Sarcasm_detection_in_Hindi/master/"
    for fname, y in (("Sarcasm_Hindi_Tweets-SARCASTIC.csv", 1.0), ("Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv", 0.0)):
        data = fetch(base + fname).decode("utf-8", "replace")
        for r in capped(csv.DictReader(io.StringIO(data)), 6000):
            add("hindi_sarcasm", re.sub(r"#\w+", " ", r.get("text") or ""), "hi", {"sarcastic": y})


def hinglish_sarcasm():
    base = f"{GH}/sahilswami96/SarcasmDetection_CodeMixed/master/Dataset/"
    tw = fetch(base + "Sarcasm_tweets.txt").decode("utf-8", "replace").split("\n\n")
    truth = [l.strip() for l in fetch(base + "Sarcasm_tweet_truth.txt").decode("utf-8", "replace").splitlines() if l.strip()]
    labels = dict(zip(truth[0::2], truth[1::2]))      # id line, YES/NO line
    for block in capped(tw, None):
        p = block.strip().split("\n", 1)
        if len(p) == 2 and p[0].strip() in labels:
            add("hinglish_sarcasm", p[1], "hi", {"sarcastic": float(labels[p[0].strip()] == "YES")})


def kaggle_csv(pattern, src, lang, fn):
    for path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if re.search(pattern, path, re.I):
            with open(path, encoding="utf-8", errors="replace", newline="") as fh:
                rd = csv.DictReader(fh)
                cols = rd.fieldnames or []
                log(f"  {src}: {path} columns {cols}")
                for r in capped(rd, None):
                    fn(r, cols)


def text_col(r, cols):
    return max(((c, r.get(c) or "") for c in cols), key=lambda kv: len(kv[1]))[1]


def ben_sarc():
    import pandas as pd
    for path in glob.glob("/kaggle/input/**/*Ben-Sarc*.xlsx", recursive=True):
        df = pd.read_excel(path)
        log("  ben_sarc columns", list(df.columns))
        lab = next(c for c in df.columns if re.search(r"polarity|label|class|sarc", str(c), re.I) and set(df[c].dropna().astype(int).unique()) <= {0, 1})
        txt = max((c for c in df.columns if c != lab and df[c].dtype == object), key=lambda c: df[c].astype(str).str.len().mean())
        for _, r in capped(df.iterrows(), None):
            add("ben_sarc", r[txt], "bn", {"sarcastic": float(int(r[lab]) == 1)})


def bengali_humor():
    def one(r, cols):
        lab = next((r[c] for c in cols if c and re.search(r"humou?r|label|class", c, re.I) and str(r[c]).strip() in ("0", "1")), None)
        if lab is not None:
            add("bengali_humor", text_col(r, cols), "bn", {"humorous": float(lab == "1")})
    kaggle_csv(r"humou?r", "bengali_humor", "bn", one)


def bad_bengali():
    for split in ("train", "validation"):
        for r in capped(hf("omar-sharif/BAD-Bengali-Aggressive-Text-Dataset", split=split), None):
            ag = r["Class"] == "AG"
            add("bad_bengali", r["cleaned"], "bn", {"aggressive": float(ag)}, "aggressive" if ag else None)


CYPHER = {"troll": ["derisive"], "sexual": ["rude"], "religious": ["aggressive"], "threat": ["aggressive"]}


def bangla_fb():
    data = fetch(f"{GH}/cypher-07/Bangla-Text-Dataset/main/dataset.csv").decode("utf-8", "replace")
    for r in capped(csv.DictReader(io.StringIO(data)), None):
        lab = (r.get("label") or "").strip()
        if lab == "not bully":
            add("bangla_fb", r["comment"], "bn", {"rude": 0.0, "aggressive": 0.0, "derisive": 0.0})
        elif lab in CYPHER:
            add("bangla_fb", r["comment"], "bn", on(*CYPHER[lab]))


EMONOBA = {"Love": "loving", "Joy": "joyful", "Surprise": "surprised", "Anger": "angry", "Sadness": "sad", "Fear": "anxious"}


def emonoba():
    for split in ("train",):
        ds = hf("khondoker/EmoNoBa", split=split)
        for r in capped(ds, None):
            text = r.get("Data") or r.get("text") or ""
            add("emonoba", text, "bn", {t: float(bool(r.get(k))) for k, t in EMONOBA.items() if k in r})


def bengali_80k():
    import pandas as pd
    raw = fetch("https://data.mendeley.com/public-files/datasets/4dnrwbxt8n/files/cad4b87e-7167-4d2c-a849-5a900e528e3a/file_downloaded")
    df = pd.read_excel(io.BytesIO(raw))
    log("  bengali_80k columns", list(df.columns))
    col = next(c for c in df.columns if re.search("bengali|bangla", str(c), re.I) and not re.search("banglish", str(c), re.I))
    lab = next(c for c in df.columns if re.search("label|emotion", str(c), re.I))
    m = {"joy": "joyful", "sadness": "sad", "anger": "angry", "disgust": "disapproving", "surprise": "surprised", "fear": "anxious"}
    for _, r in capped(df.iterrows(), None):
        t = m.get(str(r[lab]).strip().lower())
        if t:
            add("bengali_80k", r[col], "bn", {t: 1.0})


ROMAN_HI = set("hai hain nahi nhi kya kyu bhai yaar mera meri tera teri accha acha bahut bohot kuch karo raha rahi gaya hoga "
               "tum aap mujhe sahi ekdum matlab kaise abhi wala ji haan bilkul pyaar dost ghar".split())


def whatsapp():
    z = zipfile.ZipFile(io.BytesIO(fetch(
        "https://data.mendeley.com/public-files/datasets/wf5d5b2j52/files/06911351-d087-4677-af89-c483136ffe03/file_downloaded")))
    name = next(n for n in z.namelist() if n.endswith(".db"))
    path = "/tmp/whatsapp.db"
    open(path, "wb").write(z.read(name))
    con = sqlite3.connect(path)
    table = next(r[0] for r in con.execute("select name from sqlite_master where type='table'") if "CONV" in r[0].upper())
    cols = [r[1] for r in con.execute(f"pragma table_info({table})")]
    tcol = next(c for c in cols if "TEXT" in c.upper() or "MSG" in c.upper())
    seen = set()
    for (msg,) in capped(con.execute(f"select {tcol} from {table}"), None):
        msg = str(msg or "")
        if "http" in msg or len(msg) > 400 or msg in seen:        # forwards and links are mostly spam
            continue
        seen.add(msg)
        e = emoji_labels(msg)
        if e:
            words = re.findall(r"[a-z]+", msg.lower())
            add("whatsapp", msg, "hi" if sum(w in ROMAN_HI for w in words) >= 1 else "en", e=e)


for name, fn in [("isarcasm", isarcasm), ("sarc", sarc), ("talkdown", talkdown), ("mentalmanip", mentalmanip),
                 ("casino", casino), ("esconv", esconv), ("empathetic_dialogues", empathetic_dialogues),
                 ("semeval18", semeval18), ("measuring_hate", hate_measure), ("politeness", politeness), ("swda", swda),
                 ("dreaddit", dreaddit), ("dailydialog", dailydialog), ("trac2", trac2), ("hindi_sarcasm", hindi_sarcasm),
                 ("hinglish_sarcasm", hinglish_sarcasm), ("ben_sarc", ben_sarc), ("bengali_humor", bengali_humor),
                 ("bad_bengali", bad_bengali), ("bangla_fb", bangla_fb), ("emonoba", emonoba), ("bengali_80k", bengali_80k),
                 ("whatsapp", whatsapp)]:
    run(name, fn)

random.shuffle(rows)
with gzip.open(os.path.join(OUT, "tones.jsonl.gz"), "wt", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
pos = collections.Counter(t for r in rows for t, v in r["tones"].items() if v >= 1.0)
summary = {"rows": len(rows), "per_source": dict(stats), "per_language": dict(collections.Counter(r["l"] for r in rows)),
           "tone_positives": dict(pos.most_common()), "tones_without_positives": sorted(TONES - set(pos)),
           "styles": dict(collections.Counter(r["style"] for r in rows if r["style"])),
           "samples": [(r["s"], r["t"][:100], r["tones"], r["style"]) for r in rows[:40]]}
json.dump(summary, open(os.path.join(OUT, "stats.json"), "w"), ensure_ascii=False, indent=1)
log(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False, indent=1))
