# Konfiguracja trwałego cache NDVI (NeonDB + Railway)

Kroki jednorazowe, żeby włączyć trwały cache NDVI (`db_cache.py`) w produkcyjnym
środowisku lopaty. Serwis lopata sam działa na Railway, ale baza (ta sama,
której używa kret dla `farming_db`) to zewnętrzny Postgres na **NeonDB**, nie
plugin Postgres w Railway. Kod jest już wypchnięty i bezpieczny bez tego —
`LOPATA_DB_ENABLED` domyślnie `false`, więc bez poniższego serwis dalej działa
jak wcześniej (cache tylko w pamięci procesu).

## 1. Utwórz rolę w istniejącym projekcie NeonDB kreta

**Opcja A - SQL Editor w konsoli Neon (pewniejsza, ten sam styl co reszta tego
dokumentu):**

Neon Console → Twój projekt → **SQL Editor** → wykonaj (hasło wklej z
lokalnego, zignorowanego przez git pliku `.env.railway` obok tego dokumentu -
tam wygenerowałem gotowe, losowe 32-znakowe hasło; nie trzymamy sekretów w tym
committed pliku):

```sql
CREATE ROLE lopata_cache WITH LOGIN PASSWORD '<haslo z .env.railway>';
GRANT CREATE ON DATABASE <nazwa_bazy> TO lopata_cache;
```

Nazwę bazy sprawdzisz w zakładce **Databases** w konsoli Neon.

**Opcja B - zakładka Roles w konsoli Neon (Neon sam generuje haslo):**

Neon Console → projekt → branch (zwykle `main`) → **Roles** → **Add Role** →
nazwij `lopata_cache`. Neon wygeneruje i zapamięta hasło (widoczne/kopiowalne
z tego samego ekranu) - wtedy w kroku 2 użyj tego hasła zamiast tego z
`.env.railway`. Rola stworzona przez UI ma dostęp login, ale nie ma jeszcze
prawa tworzenia schematu - i tak trzeba doklikać SQL Editor i wykonać samo
`GRANT CREATE ON DATABASE ... TO lopata_cache;` z Opcji A.

Reszta (schemat `lopata`, tabela `ndvi_cache`, indeksy) tworzy się sama przy
starcie serwisu lopata — nic więcej nie trzeba klikać ani w Neon, ani w
Railway.

## 2. Ustaw zmienne serwisu lopata (w Railway)

Konfiguracja to jeden connection string zamiast osobnych pól host/port/nazwa/
użytkownik/hasło - ten sam styl co Railway's/Neon's własny `DATABASE_URL` i
kreta `SPRING_DATASOURCE_URL`. psycopg2 (sterownik lopaty) przyjmuje taki
string bezpośrednio, bez żadnego parsowania po stronie kodu.

Najprościej: Neon Console → **Connection Details** → wybierz rolę
`lopata_cache` i bazę → skopiuj gotowy connection string stamtąd (Neon sam
dokłada `?sslmode=require`, które jest tu wymagane - w przeciwieństwie do
zwykłego Postgresa w Railway, Neon **wymaga** SSL).

W Railway: serwis **lopata** → zakładka **Variables** → **Raw Editor** → wklej
zawartość lokalnego pliku `.env.railway` (obok tego dokumentu, zignorowany
przez git), podmieniając `LOPATA_DB_URL` na string skopiowany z Neona:

```env
LOPATA_DB_ENABLED=true
LOPATA_DB_URL=postgresql://lopata_cache:<haslo>@<neon-host>/<nazwa_bazy>?sslmode=require
LOPATA_DB_SCHEMA=lopata
```

Railway zrestartuje serwis lopata automatycznie po zapisaniu zmiennych.

## 3. Weryfikacja

Po restarcie, w logach serwisu lopata (Railway: zakładka **Deployments** →
**Logs**) powinna pojawić się linia:

```
lopata DB cache schema ready (lopata.ndvi_cache)
```

Jeśli zamiast tego pojawi się `Failed to initialize lopata DB cache schema`,
sprawdź `LOPATA_DB_URL` (host/port/nazwę bazy/użytkownika/hasło/`sslmode`) -
serwis dalej będzie działał (fallback do pamięci), tylko bez trwałości między
restartami.
