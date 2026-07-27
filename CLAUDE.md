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

**Znany kompromis, świadomie zaakceptowany przez użytkownika**: to poprawia strefy z dużą plamą
złego NDVI blisko cięciwy, ale na innym polu (320 "Borszyn Wielki 276/4" @2-4ha) wprowadza nowy,
mniejszy zygzak w jednej strefie (flaga `_check_sample_point_path_efficiency`: trasa 1.5-1.58x
dłuższa niż optymalna) - ten sam wzorzec "naprawa jednej strefy kosztem innej", który już wcześniej
występował w tym pliku. Dwie inne próby (lokalna naprawa "wyskoków" post-hoc; zygzak zależny od
lokalnej szerokości strefy dla stref "klepsydrowych") zostały przetestowane i odrzucone - patrz
pamięć `ndvi_sample_point_reach_cap_investigation` (2026-07-27) po szczegóły obu ślepych zaułków.

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
