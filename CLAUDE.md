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
