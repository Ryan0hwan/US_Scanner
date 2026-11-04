# 미국 주식 시스템 통합 (11종, 저장소 1개)

따로 있던 미국 시스템 11개를 저장소 하나로 합쳤습니다. 판정 조건과 디스코드 알림 모양은 각 시스템 그대로입니다 (숏 스캐너는 제외). 각 폴더의 `README.md`에 시스템별 자세한 설명이 있습니다.

| 폴더 | 시스템 | 디스코드 채널 (예시) | 시크릿 이름 |
|---|---|---|---|
| `light` | 시장 신호등 + ETF 신호등 + 위기지표 | 시장-etf-신호등체크 | `US_WEBHOOK_LIGHT` |
| `ratio` | 비교지표 (비율 32개) | 비교지표-수급이동체크 | `US_WEBHOOK_RATIO` |
| `etfvol` | ETF 거래량 (최근 3일) | etf-거래량-감시 | `US_WEBHOOK_ETFVOL` |
| `rs` | RS 대시보드 (상위 종목 + 웹페이지) | rs-rating-상위종목 | `US_WEBHOOK_RS` |
| `trend` | 미너비니 트렌드 템플릿 | 추세추종-종목 | `US_WEBHOOK_TREND` |
| `vcp` | 미너비니 VCP | 마크미너비니식-검색기 | `US_WEBHOOK_VCP` |
| `triangle` | 업센딩 트라이앵글 (v1) | 업센딩-트라이앵글-종목 | `US_WEBHOOK_TRIANGLE` |
| `bb` | 볼린저밴드 반등 | 볼린저밴드-벗어난-종목 | `US_WEBHOOK_BB` |
| `ma` | 이평선 지지 + 확률높은종목 + 장기 지지선 | 이평선지지종목 | `US_WEBHOOK_MA` |
| `delta` | 거래량 델타 (하루·5일·10일) | 볼륨델타-매수세 | `US_WEBHOOK_DELTA` |
| `pullback` | 강세주 눌림목 | 강세주눌림목 | `US_WEBHOOK_PULLBACK` |

## 공통 목록 (저장소 맨 위, 한 번만 고치면 전부 적용)

| 파일 | 쓰는 시스템 |
|---|---|
| `watchlist.txt` | 관심종목 (`티커, 회사명`). 신호등·RS·트렌드·VCP·삼각형·볼린저·이평선·델타·눌림목 |
| `etfs.txt` | ETF 105개 (`티커, 설명`). 신호등·ETF 거래량·델타 |

비율 목록은 `ratio/ratios.txt`에 있습니다.

## 자동 실행 (평일, 한국 시각, 하루 2회)

| 시각 | 실행 |
|---|---|
| **23:50** | 장중 점검 (미국 장 시작 20분 후, 서머타임 때는 1시간 20분 후). 델타·ETF 거래량은 건너뜀 (마감된 날만 보기 때문) |
| **07:10** | 장 마감 후 확정. 11개 전부 |

GitHub 사정으로 5~30분 늦을 수 있습니다. 23:50 회차는 그날 봉이 아직 진행 중이라, 장 마감 후 결과와 다를 수 있습니다. 최대 6개 시스템이 동시에 돌아서 전체가 끝나는 데 20~30분 걸립니다 (델타가 가장 오래 걸림).

## 설치 (약 20분)

### 1. 웹후크 주소 11개 모으기
디스코드 채널은 지금 쓰는 채널을 그대로 씁니다. 각 채널 ⚙ → **연동 → 웹후크** → 기존 웹후크 클릭 → **웹후크 URL 복사**. 11개 주소를 메모장에 채널 이름과 같이 적어 두세요.

### 2. 새 저장소 만들기
github.com 오른쪽 위 `+` → **New repository** → 이름 `us-scanners` → **Public** → Create repository

### 3. 파일 올리기
**uploading an existing file** → zip을 푼 `us-scanners` 폴더 **안의 내용 전부**(폴더 11개 + `watchlist.txt`, `etfs.txt`, `requirements.txt`, `README.md`)를 끌어다 놓기 → Commit changes

- 폴더째 끌어다 놓으면 폴더 구조 그대로 올라갑니다.
- `.github` 폴더는 숨김 폴더라 안 올라갈 수 있으니 4번에서 따로 만듭니다.

### 4. 자동 실행 파일
**Add file → Create new file** → 이름 칸에 `.github/workflows/main.yml` → zip 안의 같은 파일을 메모장으로 열어 전체 복사·붙여넣기 → Commit changes

### 5. 시크릿 11개 등록
**Settings → Secrets and variables → Actions → New repository secret**을 11번 반복합니다. 이름은 위 표의 **시크릿 이름**을 **직접 타이핑**하고, 값에 해당 채널의 웹후크 주소를 넣으세요.

### 6. RS 대시보드 웹페이지 켜기
**Settings → Pages** → Source: **Deploy from a branch** → Branch: **main**, 폴더: **/docs** → Save
(첫 실행 뒤 1~2분이면 `https://내아이디.github.io/us-scanners/`에 페이지가 생깁니다)

### 7. 테스트
**Actions → US Scanners → Run workflow** → system `all` → 초록 버튼. 20~30분 안에 11개 채널에 결과가 옵니다. 하나만 확인하려면 system에서 그 시스템만 고르세요.

### 8. 예전 저장소 정리 (중요)
테스트가 잘 되면 예전 미국 저장소 11개의 자동 실행을 꺼야 알림이 두 번씩 오지 않습니다. 저장소마다 **Actions → 왼쪽 워크플로 이름 → 오른쪽 위 ⋯ → Disable workflow** 하거나, 저장소를 삭제하세요. 숏 스캐너 저장소도 같은 방법으로 끄면 됩니다.

run-all을 쓰고 있다면 `us-scanners`는 자동으로 포함됩니다 (`repos.txt`에 이름을 직접 적어 두었다면 `us-scanners` 추가).

## 수동 실행 옵션

| 입력 | 설명 |
|---|---|
| system | `all` 또는 하나만 |
| force | 켜 두면 주말·장 시간과 상관없이 최근 거래일 기준으로 실행 |
| diagnose | `vcp`·`triangle`·`delta`·`pullback` 진단. 티커를 쉼표로 (`NVDA, AAPL`). 진단만 보려면 system도 그 시스템 하나만 고르세요 |

## 설정 바꾸기

각 시스템 파일 맨 위의 설정값은 예전과 같습니다 (예: `pullback/pullback.py`의 `RISK_MAX`). 시스템별 설정 표는 각 폴더의 `README.md`를 보세요. 예전 README의 설치 방법(새 저장소 만들기 등)은 이 통합 저장소에는 해당하지 않습니다.
