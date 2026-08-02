# field_zones.py - wybór punktów nakłuć (sample points)

Ten plik dokumentuje `_compute_zone_sample_points` (i jej wywołanie z `compute_field_zones`)
w `field_zones.py` - najbardziej iterowany kawałek tego serwisu. Czytaj to przed kolejną
zmianą tej funkcji, żeby nie odkrywać drugi raz tych samych ograniczeń.

## Jak to działa dzisiaj: najdłuższa linia dzieląca strefę na pół

Dla każdej strefy liczona jest jej **własna, najdłuższa linia dzieląca powierzchnię strefy
(w przybliżeniu) na pół** (`_longest_bisecting_chord`) - nie przekątna wyliczona ze wspólnej
osi PCA całego pola, jak we wcześniejszej wersji (patrz "Historia" niżej).

1. **`_longest_bisecting_chord(polygon)`.** Dla wielu kierunków (`BISECTING_CHORD_ANGLE_SAMPLES`,
   domyślnie 18, czyli co 10°) obraca wielokąt strefy tak, żeby ten kierunek stał się poziomy,
   po czym binarnie (`BISECTING_CHORD_BISECTION_ITERATIONS` = 20 kroków) szuka pozycji cięcia,
   przy której powierzchnia poniżej cięcia to dokładnie połowa całości (`Polygon.intersection`
   z prostokątem, porównanie pola - monotoniczne w pozycji cięcia, więc binary search jest
   dokładny). Odcinek tej linii wewnątrz wielokąta (`Polygon.intersection` z samą linią,
   obrócony z powrotem) to kandydat na "tę" linię dla danego kierunku; wygrywa kierunek, dla
   którego ten odcinek jest **najdłuższy** ze wszystkich sprawdzonych kierunków.
2. **Dlaczego nie przekątna PCA.** Prosta przekątna (od jednego rogu do przeciwległego wzdłuż
   wspólnej osi całego pola) zakłada kształt zbliżony do równoległoboku. Dla trójkąta to
   założenie nie ma sensu (trójkąt nie ma "przeciwległego rogu") - zgłoszony bezpośrednio
   realny przypadek: trójkątne podpole, dla którego przekątna kompletnie się nie sprawdziła.
   Zweryfikowane na syntetycznym trójkącie: `_longest_bisecting_chord` dzieli go dokładnie
   50/50 długą linią wierzchołek-do-przeciwległego-boku - dokładnie to, co dało się zauważyć
   na ręcznie narysowanej linii użytkownika.
3. **`t`/`s` względem tej linii, nie względem osi PCA.** `t(kandydat) = chord.project(Point)`
   (odległość wzdłuż linii do najbliższego miejsca na niej), `s(kandydat) =
   Point.distance(chord)` (odchylenie od linii). Cele rozłożone równo wzdłuż `[0,
   chord.length]`, ze `s_target = 0` dla każdego - bez syntetycznej amplitudy/rampy: linia już
   ma dokładnie taki kształt/zasięg, jaki wynika z geometrii strefy, nie trzeba go symulować.
4. **Reszta bez zmian względem wcześniejszej wersji**: margines `SAMPLE_POINT_MIN_DISTANCE_FROM_BOUNDARY_M`
   od granicy strefy, jednostronne unikanie najgorszych `SAMPLE_POINT_WORST_PERCENTILE`% NDVI,
   zachłanne dopasowanie najbliższego "bezpiecznego" kandydata do każdego celu z limitem kąta
   skrętu `SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES`, przebieg wydłużania końców do `t=0`/
   `t=chord.length` (dawniej do skrajnych `t` osi PCA). Wszystko to działa identycznie, tylko
   `t`/`s` liczone są względem innej linii.
5. **Trasa odwiedzania stref uproszczona z 4 do 2 kombinacji.** Każda strefa ma teraz JEDNĄ
   linię (nie dwie przekątne jak poprzednio - bisecting chord nie ma odpowiednika
   `diagonal_sign`), więc zachłanna trasa nearest-neighbor między strefami wybiera tylko
   kierunek (do przodu / odwrócona), nie linię x kierunek.

## Naprawiony bug: nieograniczone preferowanie "bezpiecznego" (nie-najgorszego-NDVI) kandydata

`_best_candidate` woli kandydata spoza najgorszych `SAMPLE_POINT_WORST_PERCENTILE`% NDVI ("bezpiecznego")
nad "niebezpiecznym" - ale wcześniej robiła to bezwarunkowo, niezależnie jak daleko trzeba było po
niego sięgnąć, dopóki cała pula bezpiecznych kandydatów w strefie nie była wyczerpana. Realny
przypadek (pole 369 "Bełcz Wielki 288" @4ha): spora, zwarta plama najgorszego NDVI leżała prawie
dokładnie na osi `_longest_bisecting_chord` tej strefy. Zachłanne dopasowanie, unikając tej plamy,
ciągnęło 10 kolejnych celów w bok, w stronę najbliższych bezpiecznych pikseli (które akurat leżały
przy własnej granicy strefy) - każdy pojedynczy krok mieścił się w limicie skrętu
`SAMPLE_POINT_MAX_TURN_ANGLE_DEGREES`, ale skumulowany dryf dawał widoczną na mapie, "zbyt regularną"
linię przyklejoną do brzegu strefy zamiast do przekątnej, z dużym skokiem z powrotem po minięciu plamy.

Naprawione: nowa stała `SAFE_PREFERENCE_MAX_REACH_MULTIPLE = 2.5` ogranicza, o ile dalej bezpieczny
kandydat może być od najbliższego niebezpiecznego, zanim przestaje być preferowany - powyżej tego
progu zachłanny dobór po prostu akceptuje bliższego "niebezpiecznego" sąsiada zamiast objeżdżać
plamę. `_best_candidate` liczy teraz NAJLEPSZEGO kandydata w OBU pulach (`_best_in_pool`), zamiast
przerywać na pierwszym trafieniu w puli bezpiecznej.

Przy wdrożeniu ten fix wprowadził nowy, mniejszy zygzak na innym polu (320 "Borszyn Wielki 276/4"
@2-4ha) - naprawiony od razu tego samego dnia przez ogólny mechanizm bezpieczeństwa opisany niżej
("Sanity-check + fallback"), nie przez dalsze strojenie tego bugu z osobna. Dwie inne próby
(lokalna naprawa "wyskoków" post-hoc; zygzak zależny od lokalnej szerokości strefy dla stref
"klepsydrowych") zostały przetestowane i odrzucone po drodze - patrz pamięć
`ndvi_sample_point_reach_cap_investigation` (2026-07-27) po szczegóły obu ślepych zaułków.

## Sanity-check + fallback: nie ufaj własnemu wynikowi zachłannego spaceru po cięciwie

Ten sam dzień, kolejny realny przypadek (pole 127 "Tworzanice 60" @4ha): zwykła, niemal idealnie
wypukła strefa (convex hull ratio 0.99999!), z realnymi kandydatami NDVI rozłożonymi na CAŁEJ
długości cięciwy (zweryfikowane bezpośrednio: pula kandydatów pokrywała t=19 do t=253 na cięciwie
o długości 270) - a mimo to wybrane punkty pokryły tylko PIERWSZĄ POŁOWĘ cięciwy (t=26 do t=139),
z `s` (odchylenie od cięciwy) rosnącym niemal liniowo do 126m, zanim algorytm po prostu przestał
próbować dotrzeć do drugiej połowy. `SAFE_PREFERENCE_MAX_REACH_MULTIPLE` tu nie pomaga - to inny
mechanizm: zachłanny spacer, ograniczony limitem kąta skrętu, potrafi zablokować się na trajektorii
odjeżdżającej od cięciwy (każdy pojedynczy krok mieści się w limicie), a próba "doskoczenia" z
powrotem do dalszej części cięciwy przy kolejnym celu wygląda jak zbyt ostry skręt względem TEJ
trajektorii, więc `_best_candidate` woli "najmniej zły" wybór blisko już-złej ścieżki niż prawdziwy
cel. Osobny mechanizm od buga naprawionego wyżej, ale ten sam rodzaj: pojedyncze kroki wyglądają
lokalnie OK, całość - nie.

Zamiast gonić kolejny wariant tego samego zachłannego-spaceru-z-limitem-skrętu, dodany został ogólny
mechanizm ochronny na końcu `_compute_zone_sample_points`: po zbudowaniu `sorted_chosen` (po obu
przebiegach doboru i przedłużaniu końców), liczone są dwie tanie miary na WŁASNYCH wybranych
punktach - `SAMPLE_POINT_MIN_CHORD_COVERAGE_FRACTION` (czy wybrane punkty pokrywają rozsądną część
`chord.length`, domyślnie min. 60%) i `SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO` (czy trasa w
zwróconej kolejności nie jest dużo dłuższa niż zachłanne najbliższy-sąsiad po tym samym zbiorze
punktów, domyślnie maks. 1.4x - te same progi/idea co własne testy `test_real_fields.py`). Jeśli
ktoś z tych progów zawiedzie, CAŁY wynik oparty na cięciwie jest odrzucany na rzecz
`_farthest_point_fallback()` (już istniejący fallback dla "brak używalnej cięciwy w ogóle") -
gorszy wizualnie (bywa zygzakiem/samoprzecinającą się pętlą zamiast czystej linii), ale realnie
pokrywa CAŁĄ strefę zamiast połowy, i to jego jedyne zadanie.

**Efekt uboczny, pozytywny**: ten sam mechanizm, bez żadnej dodatkowej zmiany, naprawił NIEZALEŻNIE
regresję na polu 320 opisaną wyżej (fallback wyłapuje dokładnie ten sam rodzaj nieefektywnej trasy).
Zweryfikowane na całym korpusie `test_real_fields.py` (9 pól x 4 wielkości docelowe, 15 pkt/strefę):
fallback uruchamia się rzadko (kilkanaście stref na >200 sprawdzonych), zero nowych regresji, dwie
wcześniej znane, nie powiązane flagi na polu 127 @1ha (90 mikro-stref) zostają bez zmian.

## Kompromis: brak gwarancji wspólnego kierunku sąsiednich stref

Wspólna oś PCA (usunięta) gwarantowała, że linie sąsiednich stref biegną w tym samym
kierunku - ważne dla wrażenia jednej ciągłej, zygzakowatej trasy przez kilka stref naraz.
Każda strefa licząca własną najdłuższą linię dzielącą niezależnie **nie ma** tej gwarancji -
kształt każdej strefy dyktuje jej własny najlepszy kierunek, niezależnie od sąsiadów. Trasa
odwiedzania stref (zachłanny nearest-neighbor, wybór kierunku per strefa) nadal robi co może,
żeby połączyć końce, ale dla mocno różniących się kształtem sąsiednich stref efekt może być
nieco mniej "gładki" niż przy wspólnej osi. Zaakceptowane świadomie: to lepsze niż sztywna
wspólna oś, która nie pasuje do żadnej z nietypowych stref (trójkąt, kształt niewypukły).

## Znany, nienaprawiony skrajny przypadek: głęboko powycinane kształty (U/klamra)

`_longest_bisecting_chord` wybiera dla każdego kierunku **najdłuższy pojedynczy ciągły**
odcinek przecięcia (`_longest_linestring_component`) spośród tego, co może być
`MultiLineString`, gdy strefa jest na tyle niewypukła, że linia cięcia wchodzi i wychodzi z
wielokąta więcej niż raz. Zweryfikowane na syntetycznym kształcie "U" (dwa ramiona połączone
wąskim mostkiem u dołu): binarny podział poprawnie znajduje pozycję cięcia dającą globalnie
50/50 powierzchni, ale gdy ta pozycja przecina wielokąt w dwóch rozłącznych miejscach (jedno
ramię + drugie ramię), funkcja bierze tylko DŁUŻSZY z tych fragmentów jako "linię" - co samo
w sobie wcale nie dzieli powierzchni po połowie (potwierdzone: 970/7229 zamiast ~4100/4100).
Realne strefy z region-growing (region growing + wygładzanie + łączenie niedomiarowych stref)
rzadko tworzą aż tak głęboko wcięte kształty (dwa odizolowane ramiona) - zgłoszony realny
przypadek (trójkąt) i typowe łagodne wybrzuszenia działają poprawnie - ale gdyby kiedyś
zgłoszono strefę w kształcie podkowy/klamry z realnie odizolowanymi punktami, to jest
źródło, nie kolejny nowy bug.

## Znaleziony i naprawiony bug: self-intersection po reprojekcji do UTM

Realny przykład (pole "Tworzanice 60", 101.89 ha, 26 podpól): jedno konkretne podpole
miało linię punktów skupioną w wąskim, prawie płaskim pasie zamiast przechodzić przez cały
kształt. Przyczyna, potwierdzona bezpośrednio na dokładnej geometrii tego podpola: wielokąt
strefy był **poprawny (valid) w lon/lat**, ale po reprojekcji do UTM
(`shp_transform(transformer.transform, geom)`) stawał się **self-intersecting (invalid)** -
różnica precyzji zmiennoprzecinkowej przy reprojekcji w okolicy prawie-stykającego się
wierzchołka (ta sama klasa problemu co `_remove_self_touching_spikes`, tylko ujawniająca
się PO reprojekcji, nie przed nią - istniejące usuwanie kolców w lon/lat tego nie łapie).

`_longest_bisecting_chord`'s `Polygon.intersection()` (przez `_safe_intersection`) na takim
invalid wielokącie **nie rzuca wyjątku** - GEOS po cichu zwraca błędny wynik zamiast
podnieść `GEOSException`, więc istniejąca obsługa wyjątków w ogóle się nie uruchamia.
Potwierdzone bezpośrednio: dla tej geometrii funkcja zwracała linię **688.7 m** przy
przekątnej bounding-boxa strefy ok. 310 m - linia sięgała kawałek poza prawdziwy wielokąt,
przez co większość celów wzdłuż niej nie miała w pobliżu żadnych prawdziwych kandydatów
NDVI, a zachłanne dopasowanie ściągnęło punkty w jedno miejsce.

Naprawione (w `_compute_zone_sample_points`, tuż przed wywołaniem
`_longest_bisecting_chord`): `utm_zone_geom = _safe_buffer0(utm_zone_geom)` - już istniejący
w tym pliku helper (`geom.buffer(0)`, standardowy trik "renoding" na drobne
self-intersections, z fallbackiem na precyzyjny grid) naprawia validność z pomijalną zmianą
pola (`4.6e-9 m²` na tym przykładzie), po czym `_longest_bisecting_chord` zwraca sensowną
linię (286.6 m, w całości wewnątrz wielokąta). **Zawsze reprojektuj geometrię strefy do UTM
przez `_safe_buffer0` przed jakąkolwiek operacją geometryczną, która zakłada poprawność
wielokąta (rotate/intersection) - walidność w lon/lat NIE gwarantuje walidności po
reprojekcji.**

## Otwarty pomysł (jeszcze niezaimplementowany): ciągłość między strefami

Użytkownik zaproponował: przy wyznaczaniu linii dla nowego podpola faworyzować taką, która
zaczyna się jak najbliżej miejsca zakończenia linii w poprzednim podpolu (w trasie
odwiedzania stref), dopóki jej długość nie spadnie poniżej konfigurowalnego progu (np. 70%)
maksymalnej możliwej długości linii dzielącej to podpole na pół. Celowo NIE
zaimplementowane od razu przy naprawie powyższego buga - ten jeden zepsuty chord (688m,
częściowo poza wielokątem) prawdopodobnie tłumaczył większość odczuwalnej utraty
ciągłości/zygzakowatości trasy na całym polu, więc warto najpierw sprawdzić na realnym
zrzucie ekranu, czy powyższa naprawa już wystarcza, zanim inwestować w tę cięższą
heurystykę. Jeśli tak - zostaje jako gotowy, przemyślany pomysł do wdrożenia w
`_longest_bisecting_chord`/trasie odwiedzania stref w `compute_field_zones` (np. próbować
kilku kątów blisko optymalnego i wybierać ten, którego chord zaczyna się najbliżej
poprzedniego końca, akceptując skrót do progu procentowego maksymalnej długości).

## Alternatywny algorytm: boustrophedon sweep (2026-08-02)

Po kolejnej sesji patchy (powyższa historia to już kilkanaście warstw poprawek na tym samym
zachłannym-marszu-po-cięciwie: reach-cap, safe-preference-cap, sanity-check+fallback, geodezyjne
ścieżki dla wygiętych stref, boustrophedon banding wewnątrz fallbacku, 2-opt, usuwanie
skrzyżowań w DWÓCH układach współrzędnych...) i dwóch nowych realnych zgłoszeń tej samej klasy
buga (pole 346 "Luboszyce Małe 23" - punkty skupione w ułamku strefy; pole 127 "Tworzanice 60" -
to samo, ale skupienie na tyle ciasne, że na mapie wygląda jak oderwany "haczyk"/pętla), dodano
**całkiem NIEZALEŻNY, drugi algorytm** zamiast kolejnej łatki: `sample_points_sweep.py`.

**Flaga wyboru**: `SAMPLE_POINT_ALGORITHM` w `field_zones.py` (`"legacy"` lub `"sweep"`),
nadpisywalna bez zmiany kodu przez zmienną środowiskową `LOPATA_SAMPLE_POINT_ALGORITHM` (ten sam
wzorzec co `ndvi_cache_ttl_seconds` w `config.py`). Dispatch siedzi w JEDNYM miejscu -
`_compute_zone_sample_points` (dawna funkcja o tej nazwie przemianowana na
`_compute_zone_sample_points_legacy`) - oba miejsca wywołania w tym pliku (early-exit
`single_zone_override` i `_select_sample_points` w `compute_field_zones`) już i tak przechodzą
przez ten jeden punkt, więc nic więcej nie trzeba było zmieniać.

**Jak działa `sample_points_sweep.py`**: zamiast JEDNEJ linii (cięciwa/ścieżka geodezyjna) na
całą strefę, dzieli strefę na RÓWNOLEGŁE PASY (rzędy) prostopadłe do krótszej osi wielokąta
strefy (z `minimum_rotated_rectangle`, nie z chmury kandydatów - ten sam "kształt strefy dyktuje
kierunek" co stary algorytm). W każdym rzędzie cele rozkładane są równomiernie wzdłuż dłuższej
osi (uwzględniając rozłączne kawałki, gdy strefa jest na tyle niewypukła, że pas przecina ją w
kilku miejscach - patrz `_evenly_spaced_targets_over_ranges`), dobierany jest najbliższy
prawdziwy kandydat NDVI do każdego celu, po czym WYNIK (nie kolejność celów) sortowany jest po
własnej pozycji na osi głównej - **to była pierwsza realna pułapka**: zachłanny dobór per-cel
jest niezależny między celami, więc kolejność przetwarzania celów NIE gwarantuje, że wybrani
kandydaci wyjdą w tej samej kolejności (jeden cel może "przegrać" najbliższego kandydata na rzecz
sąsiada) - bez tego sortowania powstawały realne samoprzecięcia. Rzędy odwiedzane są na przemian
(serpentyna), więc połączenie między rzędami to krótki, niemal prostopadły skok - stąd zakręty
~90° na końcach rzędów zamiast długich zygzaków.

**Druga pułapka**: krańce rzędów muszą być wyznaczone po KWANTYLACH prawdziwych kandydatów
(`np.quantile(s_all, ...)`), nie po równym podziale geometrycznego zasięgu strefy - realne
piksele leżą na siatce rastra (10m), więc równy podział geometryczny regularnie zostawiał rząd z
garstką/zerem kandydatów (każdy pas łapał fragment innego wiersza rastra niedopasowany do jego
własnej siatki), co zmuszało dobór do "szukania w całej puli" (patrz `_best_candidate_for_target`)
i rozbijało założenie o zgrubnej przynależności punktu do jego rzędu - to był realny, częsty
(nie brzegowy) mechanizm samoprzecięć w małych (~1ha) strefach, znaleziony i naprawiony podczas tej
samej sesji.

**Dlaczego to ma szansę być lepsze niż kolejna łatka na starym algorytmie**: cała rodzina
poprzednich bugów (patrz pamięć `ndvi_zone_boundary_invalid_data_gap`) sprowadzała się do tego, że
JEDNA wybrana linia mogła mieć świetny `t_coverage` (pokrycie WZGLĘDEM SIEBIE SAMEJ), a mimo to
obejmować tylko ułamek prawdziwego 2D kształtu strefy - bo pokrycie całej strefy było efektem
UBOCZNYM doboru punktów na jednej linii, nie czymś wymuszonym z konstrukcji. W sweep pokrycie całej
strefy wynika WPROST z konstrukcji: rzędy razem obejmują cały zakres krótszej osi, z definicji, nie
jako nadzieja że akurat tak wyjdzie.

**Wydajność**: sweep nie ma grafu widoczności/Dijkstry (ścieżki geodezyjne dla wygiętych stref w
starym algorytmie) ani wielostartowego zachłannego-najbliższy-sąsiad + wspinaczki po kącie skrętu
(stary fallback, ~2.3s/wywołanie zmierzone w poprzednich sesjach) - dobór kandydata do celu to
przeszukanie liniowe po co najwyżej kilkuset kandydatach w strefie, budowa pasów to stała liczba
operacji geometrycznych (przecięcie prostokąt x wielokąt) proporcjonalna do liczby rzędów.

**Weryfikacja 2026-08-02**: A/B na polach 346 (`@1ha`, zgłoszony bug) i 127 (`@1/4ha`, zgłoszony
bug) - wizualnie (matplotlib) sweep wyraźnie czyściejszy na obu: pole 346 legacy miało kilka stref z
zygzakami zostawiającymi połowę strefy pustą, sweep wypełnia każdą strefę systematyczną siatką
rzędów; pole 127 @4ha legacy rysował pojedynczą przekątną per strefa (często nie sięgającą rogów),
sweep rysuje zwarte równoległe rzędy pokrywające realnie całą strefę. Zero samoprzecięć po
poprawkach obu pułapek opisanych wyżej. Pełny wynik regresyjnego porównania (cały korpus
`test_real_fields.py` x oba algorytmy) - patrz commit message/pamięć sesji z 2026-08-02.

## Pełny wynik A/B na całym korpusie (2026-08-02) - sweep NIE jest jeszcze domyślny

Powyższa weryfikacja (2 zgłoszone pola) była zachęcająca, ale celowo NIE wystarczająca do zmiany
domyślnego algorytmu - `full_ab_regression.py` (scratch, nie w repo) przepuścił OBA algorytmy przez
cały korpus `test_real_fields.py` (9 pól x 5 celów: 0.5/1/2/3/4ha, `max_sample_points_per_zone=15`,
tylko ścieżka subfield-scoped, czyli ta sama co prawdziwy frontend). Wynik:

| miara | legacy | sweep |
|---|---|---|
| czas całego przebiegu | 664.2s | 156.4s |
| stref z pokryciem <50% bbox diagonal | 140 | 113 |
| segmenty skrętu >90° | 870 | **1308** |
| najgorszy pojedynczy skręt | 161.4° | **180.0°** (pełny zawrót) |
| trasy samoprzecinające się (`is_simple=False`) | 0 | **2** |
| pól/celów z flagą CHECK (zigzag/nieciągłość/etc.) | 3 | **22** |

Sweep jest ~4x szybszy i ma mniej stref o niskim pokryciu (zgodnie z zamierzeniem konstrukcji "rzędy
z definicji obejmują cały zasięg strefy") - ale ma WIĘCEJ ostrych skrętów niż legacy, nie mniej,
mimo że to był jego główny cel projektowy, plus realny 180° zawrót i 2 samoprzecinające się trasy
(legacy ma zero obu). To bezpośrednio przeczy priorytetom użytkownika (zero linii bez końca/
samoprzecięć, skręty möwliwie ≤90°) - dlatego `SAMPLE_POINT_ALGORITHM` domyślnie zostaje `"legacy"`,
`"sweep"` jest dostępny tylko przez `LOPATA_SAMPLE_POINT_ALGORITHM=sweep` do dalszej pracy, NIE jako
cichy nowy domyślny wybór.

Część flag "zigzag" (trasa >1.4x dłuższa niż zachłanne najbliższy-sąsiad) może być fałszywym
alarmem - `SAMPLE_POINT_MAX_PATH_INEFFICIENCY_RATIO` był kalibrowany pod STARY zachłanny spacer,
nie pod celowo-nie-najkrótszą systematyczną strukturę rzędów (dokładnie ta sama zastrzeżenie już
raz padło dla starego boustrophedon-bandingu w fallbacku, patrz pamięć
`ndvi_sample_point_path_algorithm` punkt 5) - ale `worst_turn=180°` i `nonsimple=2` to REALNE,
niekwestionowalne defekty samego sweep, niezależne od tego zastrzeżenia. **Następne kroki dla kogoś,
kto to podejmie**: znaleźć które konkretne strefy dają 180° zawrót / samoprzecięcie (prawdopodobnie
błąd w kolejności odwiedzania rzędów przy nietypowym kształcie strefy - patrz `_evenly_spaced_
targets_over_ranges`/serpentyna w `sample_points_sweep.py`) i naprawić PRZED rozważeniem zmiany
domyślnego algorytmu.

## Historia (dlaczego nie przekątna + wspólna oś PCA)

Wcześniejsza wersja (commity `e3e72dc`..`210eadf`) liczyła jedną wspólną oś PCA dla całego
dzielonego obszaru i dla każdej strefy budowała przekątną (liniowa rampa `s_target` od
`-amplitude` do `+amplitude`, `amplitude = 0.9 * połowa_szerokości`, z dwoma możliwymi
przekątnymi per strefa wybieranymi przez trasę odwiedzania 4 kombinacji). To dobrze
sprawdzało się dla stref zbliżonych do równoległoboku, ale kompletnie nie dla trójkąta (nie
ma przeciwległego rogu) i potrafiło zostawić punkty w odizolowanym miejscu dla niewypukłych,
wybrzuszających się stref (realny przykład: pole 3 podpola 3.11/≈3.x/3.60 ha, dolne podpole
wybrzuszające się w lewo). Diagnoza wtedy: guard na maksymalny skok w przebiegu wydłużania
rogów NIE naprawiłby tego, bo dotyczy tylko pierwszego/ostatniego punktu, a problem
występował w środku sekwencji (głównej pętli) - to właśnie doprowadziło do zastąpienia całego
podejścia najdłuższą linią dzielącą powierzchnię, opisaną wyżej.
