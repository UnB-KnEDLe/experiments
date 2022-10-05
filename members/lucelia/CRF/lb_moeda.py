from re import X


CURRENCY_SYMBOLS = {"$", "¥", "£", "€", "kr", "₽", "R$", "₹", "Rp", "₪", "zł", "Rs", "₺", "RS"}

CURRENCY_CODES = {"USD", "EUR", "CNY", "JPY", "GBP", "NOK", "DKK", "CAD", "RUB", "MXN", "ARS", "BGN",
                  "BRL", "CHF", "CLP", "CZK", "INR", "IDR", "ILS", "IRR", "IQD", "KRW", "KZT", "NGN",
                  "QAR", "SEK", "SYP", "TRY", "UAH", "AED", "AUD", "COP", "MYR", "SGD", "NZD", "THB",
                  "HUF", "HKD", "ZAR", "PHP", "KES", "EGP", "PKR", "PLN", "XAU", "VND", "GBX"}

def money_generator(doc):
    """Searches for occurrences of money patterns in text"""
    x = ''
    i = 0
    while i < len(doc):
        tok = doc[i]
        print("sim")
        if tok.text[0].isdigit():
            j = i + 1
            if i > 0 and doc[i - 1].text in (CURRENCY_CODES | CURRENCY_SYMBOLS):
                i = i - 1
                found_symbol = True
            if (j < len(doc) and doc[j].text in
                    CURRENCY_CODES | CURRENCY_SYMBOLS | {"euros", "cents", "real"}):
                j += 1
                found_symbol = True

            if found_symbol:
                print("sim")
                yield i, j, "MONEY"
                x = "Money"
            i = j
        else:
            i += 1
    return x


label = money_generator("Real R$ 10.0000,00")
print (label)