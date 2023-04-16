import requests
chart = "込"

url = "https://jisho.org/search/" + chart + "%20%23kanji"

response = requests.get(url)

if response.status_code == 200:

    data = response.text

    filewrite = open("info.txt", "w", encoding="utf-8")
    filewrite.write(data)
    filewrite.close()

    def getReadingList(unparsed):
        res = []
        tmp = unparsed.split("</a>")
        for i in range(len(tmp)-1):
            res.append(tmp[i].split(">")[1])

        return str(res)



    data = data.split("main_results")[1]
    japaneseChar = data.split("large-12 columns")[1:2]
    print(japaneseChar)

    strokeCount = data.split("kanji-details__stroke_count")[1].split("/strong")[0].split("strong>")[1].split("<")[0]
    print(strokeCount)

    grade = data.split("taught in")[1].split("</strong")[0].split("strong>")[1]
    print(grade)

    jlptLevel = data.split("JLPT level")[1].split("</")[0].split(">")[1]
    print(jlptLevel)


    english = data.split("row kanji-details--section")[1].split("kanji-details__main-meanings\">")[1]
    english = english.split("</div>")[0].strip()
    print(english)


    try:
        kunReading = data.split("<div class=\"kanji-details__main-readings\">")[1]
        kunReading = kunReading.split("<dl class=\"dictionary_entry kun_yomi\">")[1]
        kunReading = kunReading.split("<dd class=\"kanji-details__main-readings-list\" lang=\"ja\">")[1]
        kunReading = kunReading.split("</dd>")[0].strip()
        kunReading = getReadingList(kunReading)
        print(kunReading)
    except:
        print("nada")

    

    try:
        onReading = data.split("<div class=\"kanji-details__main-readings\">")[1]
        onReading = onReading.split("<dl class=\"dictionary_entry on_yomi\">")[1]
        onReading = onReading.split("<dd class=\"kanji-details__main-readings-list\" lang=\"ja\">")[1]
        onReading = onReading.split("</dd>")[0].strip()
        onReading = getReadingList(onReading)
        print(onReading)
    except:
        print("e")

   




