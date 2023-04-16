import requests

def getInfo(character):

    url = "https://jisho.org/search/" + character + "%20%23kanji"

    response = requests.get(url)

    japaneseChar, strokeCount, grade = "", "", ""
    jlptLevel, english, kunReading = "", "", ""
    onReading = ""


    if response.status_code == 200:

        data = response.text


        def getReadingList(unparsed):
            res = []
            tmp = unparsed.split("</a>")
            for i in range(len(tmp)-1):
                res.append(tmp[i].split(">")[1])

            return str(res)



        data = data.split("main_results")[1]

        japaneseChar = data.split("large-12 columns")[1:2]

        strokeCount = data.split("kanji-details__stroke_count")[1].split("/strong")[0].split("strong>")[1].split("<")[0]

        grade = data.split("taught in")[1].split("</strong")[0].split("strong>")[1]

        jlptLevel = data.split("JLPT level")[1].split("</")[0].split(">")[1]

        english = data.split("row kanji-details--section")[1].split("kanji-details__main-meanings\">")[1]
        english = english.split("</div>")[0].strip()

        try:
            kunReading = data.split("<div class=\"kanji-details__main-readings\">")[1]
            kunReading = kunReading.split("<dl class=\"dictionary_entry kun_yomi\">")[1]
            kunReading = kunReading.split("<dd class=\"kanji-details__main-readings-list\" lang=\"ja\">")[1]
            kunReading = kunReading.split("</dd>")[0].strip()
            kunReading = getReadingList(kunReading)
        except:
            print("nada")

        

        try:
            onReading = data.split("<div class=\"kanji-details__main-readings\">")[1]
            onReading = onReading.split("<dl class=\"dictionary_entry on_yomi\">")[1]
            onReading = onReading.split("<dd class=\"kanji-details__main-readings-list\" lang=\"ja\">")[1]
            onReading = onReading.split("</dd>")[0].strip()
            onReading = getReadingList(onReading)
        except:
            print("e")

    
    results = {}
    results['japaneseChar'] = japaneseChar
    results['strokeCount'] = strokeCount
    results['grade'] = grade
    results['jlptLevel'] = jlptLevel
    results['english'] = english
    results['kunReading'] = kunReading
    results['onReading'] = onReading

    return results




