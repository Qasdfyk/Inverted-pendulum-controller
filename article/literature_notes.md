# Notatki literaturowe: funkcje kosztu w MPC/NMPC dla wahadła odwróconego na wózku

Odpowiedź na komentarz promotora (dr R. Nebeluk) w `manuscript.tex`, sekcja *MPC-NO-alt control scheme* (ok. l. 399).
Stan na 2026-10-27.

**Jak weryfikowano.** Metadane każdej pozycji (autorzy, rok, tytuł, czasopismo/konferencja, tom, strony, DOI) sprawdzono w Crossref (`api.crossref.org/works/<DOI>`). Jedyny wyjątek to Kamthe2018 (PMLR nie nadaje DOI): ta pozycja jest sprawdzona na proceedings.mlr.press. Wszystkie klucze są dodane do `references.bib`.
Kolumna **Wer.** podaje, skąd pochodzi opis treści:
- **P**: przeczytany pełny tekst (fragment z funkcją kosztu),
- **A**: abstrakt lub opis wydawcy,
- **M**: tylko metadane. Zdanie opisuje temat pracy, a postać kosztu trzeba sprawdzić w PDF przed zacytowaniem w konkretnym kontekście.

---

## 1. Odpowiedź na pytanie promotora: Mills2009 i Camacho2007

### Mills, Wills, Ninness (ACC 2009), *Nonlinear model predictive control of an inverted pendulum* (Wer. P)
- Obiekt to **wahadło odwrócone na wózku** (rzeczywiste stanowisko, silnik DC sterowany PWM).
- Funkcja kosztu (eq. 3, 20–21) ma **dwa człony**: $V(u)=\sum_{k=1}^{N} x_{k+1}^T Q x_{k+1} + u_k^T R u_k$, z $Q=\mathrm{diag}(1,10,10^{-4},10^{-4})$ i $r=0{,}1$.
- Karany jest **sam sygnał sterujący $u$ (amplituda), a nie przyrost $\Delta u$**. W tekście: R „penalises input movements from the origin”.
- Uzasadnienie członu z $u$ to jedno zdanie: autorzy chcą być „economical with control action”. Wagi $Q$ dobrano jakościowo (kąt najważniejszy, potem położenie wózka, prędkości mało ważne). **W pracy nie ma badania strojenia** ani porównania wariantów kosztu.
- Ograniczenia: $|u|\le 0{,}5$, $|p|\le 0{,}7$ m (miękkie ograniczenia stanu przez zmienną slack). $N=60$ (1,5 s), $T_s=25$ ms, SQP z limitem 4 iteracji.
- **Wniosek: Mills2009 nie stosuje ani nie uzasadnia kosztu trójczłonowego $(e,\Delta u,u)$.** Ma koszt dwuczłonowy z karą na $u$ zamiast $\Delta u$, czyli taki jak LQR. Jest więc bliżej naszego członu $u^T R_\mathrm{abs} u$ niż członu $\Delta u$.
- **Uwaga do manuskryptu:** w sekcji MPC-NO (l. 372) Mills2009 jest cytowany jako źródło kosztu z $\Delta u$ (eq. `mpc_cost`). To nieścisłe: Mills karze $u$, nie $\Delta u$.

### Camacho, Bordons, *Model Predictive Control* (Wer. P dla wyd. 1, 1999)
Sprawdzono w egzemplarzu **wyd. 1** (Springer 1999, 2nd printing 2000, ISBN 3-540-76241-8, DOI 10.1007/978-1-4471-3398-8, klucz `Camacho1999`). Wydania 2 (2007, `Camacho2007`) nie mamy; jego rozdział 2 jest według Crossref odpowiednikiem tego samego rozdziału, ale tekstu wyd. 2 nie sprawdzono.
- **s. 18–19, rozdz. 2.1.2 *Objective Function*, wzór (2.5):** ogólny koszt $J=\sum\delta(j)[\hat y(t+j|t)-w(t+j)]^2+\sum\lambda(j)[\Delta u(t+j-1)]^2$. Autorzy piszą, że celem jest nadążanie wyjścia za trajektorią zadaną przy jednoczesnej karze na wysiłek sterowania ($\Delta u$). Zaraz pod wzorem (s. 19) dodają, że w części metod drugi człon się pomija, a w innych (UPC) bezpośrednio występują **wartości sygnału sterującego, a nie jego przyrosty**. **Teza 2 z sekcji 6 jest więc potwierdzona** (wyd. 1, s. 19).
- **s. 192–193, rozdz. 7.5 *1-norm*, wzór (7.16):** koszt z wartościami bezwzględnymi uchybu i $|\Delta u|$ zamiast kwadratów, motywowany tym, że zadanie sprowadza się do programowania liniowego (szybsze niż QP). To przykład alternatywnej (L1) funkcji kosztu w podręczniku. Uzasadnienie jest obliczeniowe, a nie dotyczy jakości regulacji.
- **s. 216–221, rozdz. 8.2 (Robust MPC, *Objective Functions*):** omówione normy kwadratowa, $\infty-\infty$ i 1-norm (tylko spis treści, treści nie czytano).
- **Wahadło:** rozdz. 9 *Applications* (s. 229–258) to elektrownia słoneczna, instalacja pilotowa i cukrownia. **Wahadła odwróconego w książce nie ma.**
- **Wniosek:** Camacho dopuszcza karę na $\Delta u$ (standard), wymienia wariant z karą na wartości $u$ (UPC) oraz koszt L1, ale nie analizuje kosztu trójczłonowego ($e$, $\Delta u$, $u$ razem) ani wahadła.

### Camacho, Bordons, Maestre, *Model Predictive Control*, wyd. 3 (Springer 2026) (Wer. P, EPUB) – **zalecane do cytowania**
ISBN 978-3-031-87596-0, DOI 10.1007/978-3-031-87596-0, klucz `Camacho2026`. Tekst przeszukany w całości (EPUB, bez numerów stron, więc cytować sekcję i wzór).
- **Sekcja 1.3.1 *Objective Function*, wzór (1.4):** koszt kwadratowy = błąd nadążania + $\lambda\,\Delta u^2$. W akapicie *Minimization of the Control Effort* autorzy uzasadniają człon $\Delta u$: wygładza sterowanie (kara na duże chwilowe zmiany), zwiększa efektywność energetyczną i wydłuża żywotność aktuatorów. Dalej: w części metod człon ten się pomija, a w innych w koszcie występują też bezpośrednio wartości $u$ (nie przyrosty), np. żeby sterowanie nie dochodziło do ograniczeń aktuatora i żeby poprawić stabilność przez unikanie agresywnych akcji.
- To jest **bezpośrednie uzasadnienie obu członów MPC-NO-alt** (kara na $\Delta u$ i na $u$) w podręczniku, ale ogólne, nie dla wahadła.
- **Sekcja 5.7 *Linear Programming with a 1-Norm Objective*, wzór (5.20):** koszt L1 (|uchyb|, |$\Delta u$|), motywacja obliczeniowa (LP). Rozdz. 6: normy $\infty$-$\infty$ i 1-norm w min–max MPC; rozdz. 9.1.1: 1-norm i $\infty$-norm w explicit MPC.
- **Wahadło:** słowo *pendulum* nie występuje w całej książce (rozdz. 10 *Applications* też nie).

### Nebeluk2024 (ICCMA), dla kompletności (Wer. P, PDF w repo)
- W członie $J_u$ zamienia kwadrat $\Delta u$ na 15 M-estymatorów (L1, Charbonnier, Fair, Huber, Cauchy, Tukey itd. według de Menezes 2021). **Obiekt to reaktor neutralizacji pH** (model Wienera, SISO, $N=10$, $N_u=3$), a nie wahadło.
- **Nie ma członu z karą na amplitudę $u$.** Cytowanie Nebeluk2024 jako źródła członu $u^T R_\mathrm{abs} u$ (l. 411) jest więc nieuprawnione.

---

## 2. Grupa (a): MPC/NMPC wahadła z klasycznym kosztem L2 (zachować)

| Klucz | Pełna referencja | DOI | Wer. | Jedno zdanie o funkcji kosztu |
|---|---|---|---|---|
| Mills2009 | A. Mills, A. Wills, B. Ninness, Nonlinear model predictive control of an inverted pendulum, Proc. ACC 2009, pp. 2335–2340 | 10.1109/ACC.2009.5160391 | P | NMPC (SQP, 25 ms) na rzeczywistym wahadle na wózku; koszt $x^TQx+u^TRu$ z karą na amplitudę $u$, bez $\Delta u$. |
| Magni2002 | L. Magni, R. Scattolini, K.J. Åström, Global stabilization of the inverted pendulum using model predictive control, IFAC Proc. Vol. 35(1), 2002, pp. 141–146 | 10.3182/20020721-6-ES-1901.00592 | M | Globalna stabilizacja (swing-up + stabilizacja) wahadła za pomocą MPC; klasyczna praca o nieliniowym MPC dla tego obiektu, postać kosztu do sprawdzenia. |
| Altmannshofer2012 | S. Altmannshofer, Fast suboptimal nonlinear model predictive control of an inverted pendulum, IFAC Proc. Vol. 45(17), 2012, pp. 442–447 | 10.3182/20120823-5-NL-3013.00051 | M | Szybkie, suboptymalne NMPC wahadła; temat to koszt obliczeniowy, a nie postać funkcji celu. |
| Askari2009 | M. Askari, H.A.F. Mohamed, M. Moghavvemi, S.S. Yang, Model predictive control of an inverted pendulum, TECHPOS 2009, pp. 1–4 | 10.1109/TECHPOS.2009.5412069 | M | Podstawowe liniowe MPC wahadła (cytowane m.in. przez Messikh2022). |
| Gawthrop2006 | P.J. Gawthrop, L. Wang, Intermittent predictive control of an inverted pendulum, Control Eng. Pract. 14(11), 2006, pp. 1347–1356 | 10.1016/j.conengprac.2005.09.002 | M | Przerywane (intermittent) sterowanie predykcyjne wahadła; kwadratowy wskaźnik jakości, nowością jest struktura czasowa, a nie koszt. |
| Chalupa2008 | P. Chalupa, V. Bobál, Modelling and predictive control of inverted pendulum, Proc. ECMS 2008, pp. 531–537 | 10.7148/2008-0531 | M | Modelowanie i predykcyjne sterowanie wahadła (podejście typu GPC, gdzie standardowo karany jest $\Delta u$; do sprawdzenia). |
| Messikh2017 | L. Messikh, E.H. Guechi, M.L. Benloucif, Critically damped stabilization of inverted-pendulum systems using continuous-time cascade linear MPC, J. Franklin Inst. 354(16), 2017, pp. 7241–7265 | 10.1016/j.jfranklin.2017.08.039 | M | Kaskadowe liniowe MPC w czasie ciągłym, strojone na krytycznie tłumioną odpowiedź wahadła. |
| Messikh2022 | L. Messikh, E.H. Guechi, S. Blažič, Stabilization of the cart–inverted-pendulum system using state-feedback pole-independent MPC controllers, Sensors 22(1), 2022, 243 | 10.3390/s22010243 | P | Koszt zawiera **tylko ważone błędy** (kąt, prędkość kątowa, pozycja, prędkość wózka), bez członu z $u$ ani $\Delta u$; 50 000 symulacji strojenia dwóch parametrów; przydatny kontrprzykład (patrz grupa c). |
| Bdirina2015 | E.K. Bdirina, M.S. Boucherit, R. Hadjar, M. Zineb, State constrained predictive control of cart with inverted pendulum, CEIT 2015, pp. 1–6 | 10.1109/CEIT.2015.7233021 | M | MPC wahadła z ograniczeniami stanu (np. długość toru); koszt kwadratowy. |
| Elnaggar2018 | M. Elnaggar, A. Lasheen, Explicit model predictive control for a cart inverted pendulum system, ICSC 2018, pp. 360–365 | 10.1109/ICoSC.2018.8587820 | M | Jawne (explicit, multiparametryczne) MPC wahadła; wymaga kosztu kwadratowego (lub liniowego) do rozwiązania mp-QP. |
| Tian2018 | X. Tian, H. Peng, A model predictive control approach to inverted pendulum system based on RBF-ARX model, CCC 2018, pp. 3559–3564 | 10.23919/ChiCC.2018.8483146 | M | MPC z modelem RBF-ARX dla wahadła; nowością jest model, koszt kwadratowy. |
| He2023 | J. He, Y. Li, Z. Wei, Z. Huang, Gain-scheduled MPC for cart–inverted-pendulum with friction and disturbances, Appl. Sci. 13(24), 2023, 13080 | 10.3390/app132413080 | M | MPC z harmonogramowaniem wzmocnień, z tarciem i zakłóceniami; świeży przykład klasycznego podejścia L2. |
| Jezierski2017 | A. Jezierski, J. Mozaryn, D. Suski, A comparison of LQR and MPC control algorithms of an inverted pendulum, AISC 577, 2017, pp. 65–76 | 10.1007/978-3-319-60699-6_8 | M (już w bib) | Porównanie LQR i MPC na wahadle; klasyczny koszt kwadratowy. |
| Varghese2017 | E.S. Varghese, A.K. Vincent, V. Bagyaveereswaran, Optimal control of inverted pendulum system using PID controller, LQR and MPC, IOP Conf. Ser. MSE 263(5), 2017, 052007 | 10.1088/1757-899X/263/5/052007 | M (już w bib) | Porównanie PID/LQR/MPC na wahadle; klasyczny koszt kwadratowy. |
| Gros2020 | S. Gros, M. Zanon, R. Quirynen, A. Bemporad, M. Diehl, From linear to nonlinear MPC: bridging the gap via the real-time iteration, Int. J. Control 93(1), 2020, pp. 62–80 | 10.1080/00207179.2016.1222553 | M | Tutorial RTI dla NMPC; **nie potwierdzono**, że zawiera przykład wahadła na wózku. Cytować tylko jako źródło metody RTI. |

---

## 3. Grupa (b): alternatywne / niestandardowe funkcje kosztu dla wahadła (najważniejsza)

**Ogólny wniosek z poszukiwań:** nie znaleziono pracy, która dla wahadła na wózku zastępuje człon $\Delta u$ M-estymatorami (Huber, Cauchy, Tukey itp.), stosuje min-max MPC na rzeczywistym stanowisku albo ekonomiczne MPC. Wyszukiwanie w Crossref i w sieci dla haseł *L1 / Huber / min-max / economic / energy-based cost + cart-pole MPC* nie dało takich trafień. Niestandardowe koszty dla tego obiektu występują w trzech nurtach: (i) teoria NMPC (koszty trygonometryczne/L1 dobrane do stabilności), (ii) uczenie ze wzmocnieniem i sample-based MPC (koszty nasycające i trygonometryczne), (iii) swing-up (koszt tylko końcowy lub energia). **To jest luka, którą zapełnia nasz artykuł, i warto to wprost napisać we wstępie.**

| Klucz | Pełna referencja | DOI | Wer. | Jedno zdanie o funkcji kosztu |
|---|---|---|---|---|
| Grune2010 | L. Grüne, J. Pannek, M. Seehafer, K. Worthmann, Analysis of unconstrained nonlinear MPC schemes with time varying control horizon, SIAM J. Control Optim. 48(8), 2010, pp. 4938–4962 | 10.1137/090758696 | P | Dla **wahadła na wózku** stosuje (a) w wersji liniowej koszt w **normie L1** $\|Qx\|_1+\|Ru\|_1$, (b) w wersji nieliniowej **niekwadratowy koszt trygonometryczny** (człony $\sin^2 x_1$, $(1-\cos x_1)(1+\cos x_2)^2$ itd., mała kara $10^{-4}u^2$); pokazuje, że wybór kosztu i wagi końcowej decyduje o minimalnym horyzoncie stabilizującym. |
| GrunePannek2017 | L. Grüne, J. Pannek, Nonlinear Model Predictive Control: Theory and Algorithms, 2nd ed., Springer, 2017 | 10.1007/978-3-319-46024-6 | M | Monografia NMPC; wahadło na wózku jest jednym z przykładów przewodnich, a dobór kosztu etapowego to jeden z tematów książki (konkretny przykład/strona do wskazania po sprawdzeniu w egzemplarzu). |
| Muller2017 | M.A. Müller, K. Worthmann, Quadratic costs do not always work in MPC, Automatica 82, 2017, pp. 269–277 | 10.1016/j.automatica.2017.04.058 | A | Pokazuje, że dla pewnych układów nieliniowych koszt kwadratowy nie daje stabilności MPC bez warunków końcowych, i proponuje koszty niekwadratowe; ogólne uzasadnienie szukania alternatyw dla L2. |
| Jung2004 | S. Jung, J.T. Wen, Nonlinear model predictive control for the swing-up of a rotary inverted pendulum, J. Dyn. Syst. Meas. Control 126(3), 2004, pp. 666–673 | 10.1115/1.1789541 | A | NMPC dla swing-upu wahadła **rotacyjnego** (Furuta), które minimalizuje **tylko błąd na końcu horyzontu** zamiast pełnego kosztu sumowanego; eksperyment. |
| Deisenroth2015 | M.P. Deisenroth, D. Fox, C.E. Rasmussen, Gaussian processes for data-efficient learning in robotics and control, IEEE TPAMI 37(2), 2015, pp. 408–423 | 10.1109/TPAMI.2013.218 | P | PILCO: dla **cart-pole swing-up** używa **nasycającego kosztu** $1-\exp(-d^2/2\sigma_c^2)$ (d: odległość końca wahadła od celu), uzasadnionego automatyczną eksploracją; to nie MPC, ale wzorcowy niekwadratowy koszt dla tego obiektu. |
| Kamthe2018 | S. Kamthe, M.P. Deisenroth, Data-efficient reinforcement learning with probabilistic model predictive control, Proc. AISTATS 2018, PMLR 84, pp. 1701–1710 | (brak DOI; proceedings.mlr.press/v84/kamthe18a) | P | **MPC** z modelem GP dla cart-pole swing-up z tym samym **nasycającym kosztem** co PILCO oraz ograniczeniami szansowymi (ściana na torze). |
| Williams2015 | G. Williams, A. Aldrich, E.A. Theodorou, Model predictive path integral control using covariance variable importance sampling, arXiv:1509.01149, 2015 | 10.48550/arXiv.1509.01149 | P | MPPI (sample-based MPC) dla **cart-pole swing-up** z kosztem stanu $p^2+500(1+\cos\theta)^2+\dot\theta^2+\dot p^2$ (człon trygonometryczny) i $R=1$ na sterowanie. |
| Williams2017 | G. Williams, A. Aldrich, E.A. Theodorou, Model predictive path integral control: from theory to parallel computation, J. Guid. Control Dyn. 40(2), 2017, pp. 344–357 | 10.2514/1.G001921 | M | Wersja czasopismowa MPPI; koszt przy cart-pole potwierdzono tylko w preprincie (Williams2015), więc do cytowania w tym kontekście lepiej użyć preprintu albo sprawdzić PDF JGCD. |
| Nakatani2019 | S. Nakatani, H. Date, Swing up control of inverted pendulum on a cart with collision by Monte Carlo model predictive control, SICE 2019, pp. 1050–1055 | 10.23919/SICE.2019.8859912 | A | Bezgradientowe Monte Carlo MPC dla swing-upu wahadła na wózku z kolizjami (odbojniki): koszt nie musi być różniczkowalny, a sterowanie jest nieciągłe. |
| Astrom2000 | K.J. Åström, K. Furuta, Swinging up a pendulum by energy control, Automatica 36(2), 2000, pp. 287–295 | 10.1016/S0005-1098(99)00140-5 | A | Klasyczny swing-up oparty na **energii** (cel: $E\to E_\mathrm{up}$); to nie MPC, ale źródło kosztów „energetycznych” stosowanych potem w NMPC swing-upu. |

---

## 4. Grupa (c): rola i korzyści członów z karą na $u$ i/lub $\Delta u$

**Ogólny wniosek:** nie znaleziono pracy, która **dla wahadła** systematycznie bada wpływ członu $\Delta u$ i członu $u$ (oba razem) na jakość regulacji. Prace o wahadle albo karzą tylko $u$ (Mills2009), albo tylko $\Delta u$ (podejścia GPC), albo żadnego z nich (Messikh2022). Argumenty za członami pochodzą z ogólnej literatury MPC:

| Klucz | Pełna referencja | DOI | Wer. | Co mówi o członach $u$ / $\Delta u$ |
|---|---|---|---|---|
| Mills2009 | j.w. | 10.1109/ACC.2009.5160391 | P | Jedyne znalezione uzasadnienie dla wahadła: kara na $u$, żeby być „economical with control action”; $r=0{,}1$ przy $q_\theta=10$; bez badania wrażliwości. |
| Messikh2022 | j.w. | 10.3390/s22010243 | P | Kontrprzykład: rezygnuje z kary na sterowanie, a „reduced control effort” osiąga przez strojenie wag błędów i horyzontu; zawiera **dokładne badanie strojenia** (50 000 symulacji, siatki parametrów, margines wzmocnienia), ale bez członów $u$/$\Delta u$. |
| Camacho1999 | E.F. Camacho, C. Bordons, Model Predictive Control, Springer, 1999 (wyd. 1) | 10.1007/978-1-4471-3398-8 | P | Wzór (2.5), s. 19: koszt z $\lambda\,\Delta u^2$ jako postać ogólna; pod wzorem: w UPC w koszcie występują wartości $u$, a nie przyrosty. Rozdz. 7.5 (s. 192): koszt L1 (LP). |
| Camacho2007 | E.F. Camacho, C. Bordons, Model Predictive Control, 2nd ed., Springer, 2007 | 10.1007/978-0-85729-398-5 | M | Wydanie 2; tekstu nie sprawdzono. Cytować `Camacho1999` z numerem strony albo sprawdzić stronę w wyd. 2. |
| Tatjewski2007 | P. Tatjewski, Advanced Control of Industrial Processes: Structures and Algorithms, Springer, 2007 | 10.1007/978-1-84628-635-3 | M | Podręcznik (także cytowany w Nebeluk2024 jako [1]) z klasyczną postacią kosztu: błędy + $\lambda\,\Delta u^2$. |
| Shridhar1997 | R. Shridhar, D.J. Cooper, A tuning strategy for unconstrained SISO model predictive control, Ind. Eng. Chem. Res. 36(3), 1997, pp. 729–746 | 10.1021/ie9604280 | A | Analityczne reguły doboru współczynnika tłumienia ruchów (move suppression, kara na $\Delta u$) i horyzontów; klasyczne źródło o roli $\lambda$. |
| Garriga2010 | J.L. Garriga, M. Soroush, Model predictive control tuning methods: a review, Ind. Eng. Chem. Res. 49(8), 2010, pp. 3505–3515 | 10.1021/ie900323c | A | Przegląd metod strojenia MPC, w tym wag $\Delta u$ i $u$ oraz ich wpływu na odporność i agresywność sterowania. |
| Nebeluk2021Algorithms | R. Nebeluk, M. Ławryńczuk, Tuning of multivariable model predictive control for industrial tasks, Algorithms 14(1), 2021, 10 | 10.3390/a14010010 | M | Strojenie wag MPC (w tym kary $\Delta u$) dla zadań wielowymiarowych; przykład systematycznego strojenia z grupy promotora. |
| Rao1999 | C.V. Rao, J.B. Rawlings, Steady states and constraints in model predictive control, AIChE J. 45(6), 1999, pp. 1266–1278 | 10.1002/aic.690450612 | P | Wzór (11), s. 1269: koszt regulatora $\sum w^TC^TQCw + v^TRv + \Delta v^TS\Delta v$, czyli **trzy człony**: wyjście ($Q$), wejście względem $u_{ss}$ ($R$) i przyrost wejścia ($S$). Kara na $u$ liczona jest od wartości ustalonej $u_{ss}$; dla wahadła w górnym położeniu $u_{ss}=0$, więc odpowiada to członowi $u^TR_\mathrm{abs}u$. |
| Muske1993 | K.R. Muske, J.B. Rawlings, Linear model predictive control of unstable processes, J. Process Control 3(2), 1993, pp. 85–96 | 10.1016/0959-1524(93)80004-U | P | MPC dla **obiektów niestabilnych w otwartej pętli**. Wzór (2), s. 86: koszt $\sum (y-y_s)^TQ_y(y-y_s) + (u-u_s)^TR(u-u_s) + \Delta u^TS\Delta u$ (trzy człony, wersja skończona: wzór (5)). Twierdzenie 1 (s. 87): stabilność nominalna dla $R>0$, $Q\ge0$, $S\ge0$. Dobre źródło dla postaci kosztu MPC-NO-alt. |
| Pannocchia2003 | G. Pannocchia, J.B. Rawlings, Disturbance models for offset-free model-predictive control, AIChE J. 49(2), 2003, pp. 426–437 | 10.1002/aic.690490213 | A | MPC bez uchybu ustalonego: kara na $u$ musi być liczona względem $u_{ss}$, inaczej powstaje offset. **Argument dla nas:** w górnym położeniu równowagi wahadła $u_{ss}=0$, więc człon $u^TR_\mathrm{abs}u$ nie wprowadza uchybu ustalonego (dopóki nie ma stałego zakłócenia lub tarcia statycznego). |
| Domanski2021Access | P.D. Domański, M. Ławryńczuk, Impact of MPC embedded performance index on control quality, IEEE Access 9, 2021, pp. 24787–24795 | 10.1109/ACCESS.2021.3057674 | A | Pokazuje, że postać wskaźnika wbudowanego w MPC (np. kwadratowy vs. inny) zmienia ocenę jakości wieloma kryteriami; bezpośrednio wspiera naszą metodologię porównania. |
| Domanski2024 | P.D. Domański, Energy-aware multicriteria control performance assessment, Energies 17(5), 2024, 1173 | 10.3390/en17051173 | M | Ocena jakości regulacji z uwzględnieniem energii/wysiłku sterowania; uzasadnia mierzenie zużycia „energii sterowania” obok błędów. |

---

## 5. Grupa (d): prace o funkcjach kosztu (Nebeluk/Ławryńczuk i pokrewne)

| Klucz | Pełna referencja | DOI | Wer. | Jedno zdanie |
|---|---|---|---|---|
| Lawrynczuk2021 | M. Ławryńczuk, R. Nebeluk, Computationally efficient nonlinear MPC using the L1 cost-function, Sensors 21(17), 2021, 5835 | 10.3390/s21175835 | A | NMPC z normą **L1** w członie błędów; |·| aproksymowany siecią neuronową i linearyzowany, co daje zadanie QP; L1 zwykle daje lepszą jakość niż L2. |
| Lawrynczuk2023 | M. Ławryńczuk, R. Nebeluk, Beyond the quadratic norm: computationally efficient constrained nonlinear MPC using a custom cost function, ISA Trans. 134, 2023, pp. 336–356 | 10.1016/j.isatra.2022.09.010 | M | Rozszerzenie na dowolną (custom) funkcję kosztu w członie błędów z ograniczeniami, z wydajną implementacją. |
| Nebeluk2022MED | R. Nebeluk, M. Ławryńczuk, Fast NMPC using a custom cost-function: preliminary results, MED 2022, pp. 13–18 | 10.1109/MED54222.2022.9837207 | M | Wstępne wyniki szybkiego NMPC z niestandardową funkcją kosztu. |
| Nebeluk2023IFAC | R. Nebeluk, M. Ławryńczuk, NMPC with L1 cost-function using neural networks for multivariable processes, IFAC-PapersOnLine 56(2), 2023, pp. 1591–1596 | 10.1016/j.ifacol.2023.10.1859 | M | NMPC-L1 z aproksymacją neuronową dla procesów wielowymiarowych (IFAC World Congress 2023). |
| Nebeluk2023LNNS | R. Nebeluk, M. Ławryńczuk, On the choice of the cost function for NMPC: a multi-criteria evaluation, LNNS, Springer, 2023, pp. 361–371 | 10.1007/978-3-031-35170-9_34 | M | Wielokryterialna ocena wyboru funkcji kosztu NMPC; bardzo bliska metodologii naszego artykułu. |
| Nebeluk2024 | R. Nebeluk, Exploring the benefits of alternative cost functions for NMPC, ICCMA 2024, pp. 60–65 | 10.1109/ICCMA63715.2024.10843946 | P | 15 M-estymatorów zamiast $\Delta u^2$ w członie $J_u$; reaktor pH; ocena wskaźnikami MSE, MAE, KVT i innymi; **bez członu $u$, bez wahadła**. |
| Nebeluk2022Energies | R. Nebeluk, M. Ławryńczuk, Fast MPC of PEM fuel cell system using the L1 norm, Energies 15(14), 2022, 5157 | 10.3390/en15145157 | M | MPC-L1 dla ogniwa paliwowego PEM; zastosowanie normy L1 do obiektu energetycznego. |
| Nebeluk2021Access | R. Nebeluk, M. Ławryńczuk, Computationally simple NMPC algorithm for vehicle obstacle avoidance with minimization of fuel utilization, IEEE Access 9, 2021, pp. 17296–17311 | 10.1109/ACCESS.2021.3054675 | A | Dodatkowy, „ekonomiczny” człon kosztu (zużycie paliwa) w NMPC pojazdu; przykład członu kosztu innego niż błąd i $\Delta u$. |
| Nebeluk2026 | R. Nebeluk, Development of state-space MPC with L1 cost-function and deep state-space neural models, Commun. Nonlinear Sci. Numer. Simul. 157, 2026, 109742 | 10.1016/j.cnsns.2026.109742 | M | Najnowsza praca: MPC w przestrzeni stanu z kosztem L1 i głębokimi modelami neuronowymi (warto zacytować jako najświeższą). |
| Domanski2020ACC | P.D. Domański, M. Ławryńczuk, Multi-criteria control performance assessment method for multivariate MPC, ACC 2020, pp. 1968–1973 | 10.23919/ACC45564.2020.9147815 | M | Metoda wielokryterialnej oceny jakości MPC. |
| Domanski2021Access | (patrz grupa c) | 10.1109/ACCESS.2021.3057674 | A | j.w. |
| Domanski2024 | (patrz grupa c) | 10.3390/en17051173 | M | j.w. |
| deMenezes2021 | D.Q.F. de Menezes, D.M. Prata, A.R. Secchi, J.C. Pinto, A review on robust M-estimators for regression analysis, Comput. Chem. Eng. 147, 2021, 107254 | 10.1016/j.compchemeng.2021.107254 | A | Przegląd M-estymatorów (źródło definicji ρ-funkcji użytych w Nebeluk2024 i w naszym badaniu). |
| Muller2017 | (patrz grupa b) | 10.1016/j.automatica.2017.04.058 | A | Koszt kwadratowy nie zawsze działa w MPC. |
| Karamanakos2018 | P. Karamanakos, T. Geyer, R. Kennel, On the choice of norm in finite control set MPC, IEEE Trans. Power Electron. 33(8), 2018, pp. 7105–7117 | 10.1109/TPEL.2017.2756092 | A | Porównanie norm L1, L2 i L∞ w FCS-MPC (energoelektronika); L2 zwykle daje mniejsze zniekształcenia, a L1 zachowuje się inaczej przy przełączaniu. |
| Feher2020 | M. Fehér, O. Straka, V. Šmídl, MPC of electric drive system with L1-norm, Eur. J. Control 56, 2020, pp. 242–253 | 10.1016/j.ejcon.2020.01.008 | M | MPC napędu elektrycznego z normą L1 (cytowane w Nebeluk2024). |
| Lawrynczuk2014 | M. Ławryńczuk, Computationally Efficient Model Predictive Control Algorithms: A Neural Network Approach, Springer, 2014 | 10.1007/978-3-319-04229-9 | M | Monografia: wydajne algorytmy MPC (linearyzacja, sieci neuronowe); tło dla NMPC z niestandardowym kosztem. |

---

## 6. Które twierdzenia w akapicie `manuscript.tex` l. 400–411 mają wsparcie w literaturze

| # | Twierdzenie (skrót) | Ocena | Komentarz / proponowane źródła |
|---|---|---|---|
| 1 | „Klasyczny dwuczłonowy koszt kwadratowy (błąd + wysiłek sterowania) jest domyślnym wyborem w NMPC, np. Mills2009” | **Wspierane** | Mills2009 tak, ale karze **$u$**, nie $\Delta u$. Dodać: Magni2002, Altmannshofer2012, Jezierski2017, Elnaggar2018, He2023, Bdirina2015. **Uwaga:** l. 372 przypisuje Millsowi koszt z $\Delta u$, co jest nieścisłe. |
| 2 | „Camacho2007 dopuszcza człony move-suppression w $\Delta u$ i opcjonalnie karę na amplitudę $u$” | **Wspierane** (wyd. 1) | Sprawdzone w Camacho1999, s. 19 (pod wzorem 2.5): koszt ogólny ma człon $\Delta u$, a w UPC występują bezpośrednio wartości $u$. Cytować `Camacho1999` (sprawdzone wydanie). |
| 3 | „…ale nie traktuje wahadła jako studium przypadku kosztu trójczłonowego” | **Wspierane** | Spis rozdziałów (Crossref) nie zawiera wahadła; rozdz. 12 (Applications) nieprzeczytany. Bezpieczniej: „does not consider the inverted pendulum”. |
| 4 | „Alternatywne (niekwadratowe) normy dla NMPC są przeglądane i porównywane w Nebeluk2024” | **Częściowo / przesadzone** | Nebeluk2024 porównuje 15 M-estymatorów **w członie $\Delta u$** na reaktorze pH; to nie przegląd norm. Lepiej: „M-estimator-based penalties on $\Delta u$ were compared in [Nebeluk2024], and L1/custom costs on the error term in [Lawrynczuk2021, Lawrynczuk2023, Nebeluk2023IFAC]; see also [Muller2017, Karamanakos2018]”. |
| 5 | „Człon $\Delta u$ wynika z potrzeby gładkich sił wózka, które ograniczają jerk i zużycie aktuatora na krótkim torze” | **Brak źródła dla wahadła** | Ogólnie wspierają to Camacho2026 (sekcja 1.3.1: człon $\Delta u$ wygładza sterowanie, poprawia efektywność energetyczną i żywotność aktuatorów), Shridhar1997 i Garriga2010. Dla wahadła nie znaleziono pracy, która by to pokazała; sformułować jako hipotezę i pokazać na naszych wynikach (np. TV(u), $\sum\Delta u^2$). |
| 6 | „Człon $u^TR_\mathrm{abs}u$ bezpośrednio zmniejsza energię i siłę szczytową potrzebne do utrzymania górnej równowagi” | **Częściowo** | Camacho2026 (sekcja 1.3.1): kara na wartości $u$ chroni przed nasyceniem aktuatora i poprawia stabilność przez unikanie agresywnych akcji. Mills2009 uzasadnia karę na $u$ „ekonomicznością” sterowania (bez danych). Dodatkowy argument: $u_{ss}=0$ w górnym położeniu, więc kara na $u$ nie wprowadza uchybu (Pannocchia2003, Rao1999). Redukcję energii i siły szczytowej trzeba **wykazać własnymi wynikami** (Domanski2024 jako źródło metryk energetycznych). |
| 7 | „…oba efekty są niezbędne przy porównywaniu regulatorów predykcyjnych przy tych samych horyzontach i wagach stanu” | **Niewspierane** | Teza retoryczna, bez źródła; usunąć albo przeformułować jako cel naszego badania. |
| 8 | l. 411: „MPC-NO-alt rozszerza koszt o karę na amplitudę [Camacho2007, Nebeluk2024]” | **Nebeluk2024: niewspierane** | Nebeluk2024 nie ma członu $u$. Zastąpić przez Mills2009 (kara na $u$ dla wahadła) + Muske1993/Rao1999 (koszt trójczłonowy z $Q$, $R$, $S$; sprawdzone) + Camacho2026, sekcja 1.3.1 (kara na wartości $u$; sprawdzone). |

**Sugerowana luka do podkreślenia we wstępie:** dla wahadła na wózku literatura MPC używa niemal wyłącznie kosztu kwadratowego (grupa a). Niestandardowe koszty pojawiają się tylko w teorii stabilności NMPC (Grune2010: L1 i koszt trygonometryczny), w RL/sample-based MPC (Deisenroth2015, Kamthe2018, Williams2015: koszty nasycające/trygonometryczne) i w swing-upie (Jung2004, Astrom2000). Nie znaleziono pracy, która stosuje M-estymatory w członie $\Delta u$ dla tego obiektu ani systematycznie bada razem człony $u$ i $\Delta u$.

---

## 7. Poprawki w `references.bib` (istniejące wpisy)

- **Mills2009**: DOI było błędne (`10.1109/ACC.2009.5160600` to praca Ding, *Reduced order observer design…*). Poprawione na **`10.1109/ACC.2009.5160391`** (sprawdzone w Crossref).
- **Camacho2007**: DOI było błędne (`10.1007/978-1-84628-615-5` to Isidori, *Nonlinear Control Systems*, 1995). Poprawione na **`10.1007/978-0-85729-398-5`**; dodana seria.
- Do decyzji (nie zmieniano): styl `spmpsci` zamienia tytuły na małe litery, więc „LQR”, „MPC”, „PID” w tytułach Jezierski2017, Varghese2017 i Prasad2014 wychodzą jako „lqr”, „mpc”. Trzeba je ująć w `{...}` (w nowych wpisach już to zrobiono).

## 8. Kandydaci niezweryfikowani (nie cytować bez sprawdzenia)

- ~~Camacho2007, rozdz. 2: zdanie o UPC~~ sprawdzone w wyd. 1 (Camacho1999, s. 19).
- ~~Rao1999 / Muske1993~~ sprawdzone: oba mają koszt trójczłonowy $(Q,R,S)$.
- Gros2020: czy zawiera przykład wahadła na wózku.
- Williams2017 (JGCD): czy zawiera przykład cart-pole (potwierdzono tylko w preprincie Williams2015).
- GrunePannek2017: numer przykładu z wahadłem i kosztem trygonometrycznym.
- Przykłady MathWorks (*MPC Control of an Inverted Pendulum on a Cart*; zwiększanie wagi „MV rate” z 0,1 do 1 dla odporności) to dokumentacja, a nie literatura recenzowana; najwyżej jako przypis.
