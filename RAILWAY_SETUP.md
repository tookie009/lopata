# Konfiguracja trwałego cache NDVI na Railway

Kroki jednorazowe, żeby włączyć trwały cache NDVI (`db_cache.py`) w produkcyjnym
środowisku lopaty na Railway. Kod jest już wypchnięty i bezpieczny bez tego —
`LOPATA_DB_ENABLED` domyślnie `false`, więc bez poniższego serwis dalej działa
jak wcześniej (cache tylko w pamięci procesu).

## 1. Utwórz rolę i schemat w istniejącym Postgresie kreta

Otwórz zakładkę **Query** pluginu Postgres kreta w Railway (ten sam, którego
używa `farming_db`) i wykonaj (hasło wklej z lokalnego, zignorowanego przez
git pliku `.env.railway` obok tego dokumentu - tam wygenerowałem gotowe,
losowe 32-znakowe hasło; nie trzymamy sekretów w tym committed pliku):

```sql
CREATE ROLE lopata_cache WITH LOGIN PASSWORD '<haslo z .env.railway>';
GRANT CREATE ON DATABASE railway TO lopata_cache;
```

Uwaga: zmień `railway` na faktyczną nazwę bazy, jeśli Railway nazwał ją inaczej
(widoczne w zakładce **Variables** pluginu Postgres, zmienna `PGDATABASE`).
Reszta (schemat `lopata`, tabela `ndvi_cache`, indeksy) tworzy się sama przy
starcie serwisu lopata — nic więcej nie trzeba klikać.

## 2. Ustaw zmienne serwisu lopata

Konfiguracja to jeden connection string zamiast osobnych pól host/port/nazwa/
użytkownik/hasło - ten sam styl co Railway's własny `DATABASE_URL` per plugin
Postgres i kreta `SPRING_DATASOURCE_URL`. psycopg2 (sterownik lopaty) przyjmuje
taki string bezpośrednio, bez żadnego parsowania po stronie kodu.

W Railway: serwis **lopata** → zakładka **Variables** → **Raw Editor** → wklej
zawartość lokalnego pliku `.env.railway` (obok tego dokumentu, zignorowany
przez git):

```env
LOPATA_DB_ENABLED=true
LOPATA_DB_URL=postgresql://lopata_cache:<haslo z .env.railway>@${{Postgres.PGHOST}}:${{Postgres.PGPORT}}/${{Postgres.PGDATABASE}}
LOPATA_DB_SCHEMA=lopata
```

(`Postgres` zastąp faktyczną nazwą pluginu widoczną w Twoim projekcie, jeśli
jest inna - to działa tylko gdy lopata i kret są w tym samym projekcie Railway,
tak że `${{Postgres.PGHOST}}`-owe referencje w ogóle się rozwiążą). Jeśli nie
są w tym samym projekcie, wklej zamiast referencji gotowe wartości `PGHOST`/
`PGPORT`/`PGDATABASE` z zakładki **Variables** samego pluginu Postgres wprost
do stringa.

Railway zrestartuje serwis lopata automatycznie po zapisaniu zmiennych.

## 3. Weryfikacja

Po restarcie, w logach serwisu lopata (zakładka **Deployments** → **Logs**)
powinna pojawić się linia:

```
lopata DB cache schema ready (lopata.ndvi_cache)
```

Jeśli zamiast tego pojawi się `Failed to initialize lopata DB cache schema`,
sprawdź poprawność `LOPATA_DB_HOST/PORT/NAME/USER/PASSWORD` - serwis dalej
będzie działał (fallback do pamięci), tylko bez trwałości między restartami.
