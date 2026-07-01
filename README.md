# Subtitle Extractor

Program do wyciągania napisów z filmów (OCR), ich edycji, tłumaczenia oraz nauki języka.

## Jak to odpalić? (krok po kroku)

1. **Zainstaluj Python 3.11** ze strony python.org.
   - **WAŻNE:** Podczas instalacji zaznacz kwadracik "Add Python 3.11 to PATH".
2. **Pobierz ten projekt** (np. jako plik ZIP i wypakuj go).
3. **Zainstaluj wymagane biblioteki:**
   - Otwórz folder z projektem.
   - Kliknij prawym przyciskiem w puste miejsce w folderze i wybierz "Otwórz w Terminalu".
   - Wpisz: `pip install -r requirements.txt` i naciśnij Enter.
4. **Odpal program:**
   - Po prostu kliknij dwukrotnie plik `gui.bat`.

## Co potrafi ten program?

- **Ekstrakcja (OCR):** Wyciągasz napisy "wtopione" w film. Program sam je czyta z obrazu.
- **Edytor:** Możesz poprawiać napisy, zmieniać czas ich wyświetlania, dodawać kolory (ASS).
- **Tłumacz:** Tłumaczysz gotowe napisy na polski lub inne języki.
- **Słownik i nauka:**
    - **Słownik:** Wyszukujesz słowa (chińskie, pinyin, angielskie) – program podaje znaczenie, HSK i inne szczegóły.
    - **Lista słówek:** Możesz zaznaczać słówka z napisów i zapisywać je do swojej własnej listy, żeby później łatwo je powtórzyć.

## Ważne uwagi:
- **VLC:** Program korzysta z VLC Media Player do odtwarzania wideo, upewnij się, że masz go zainstalowanego w systemie.
- **Tłumaczenia:** Jeśli chcesz tłumaczyć przez DeepL, OpenAI czy Google, w ustawieniach programu będziesz musiał wpisać swój własny klucz API.
- **Wersja Pythona:** Program działa najlepiej na Pythonie 3.11.
