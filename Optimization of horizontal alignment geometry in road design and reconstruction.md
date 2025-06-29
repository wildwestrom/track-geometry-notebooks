# Optimization of horizontal alignment geometry in road design and reconstruction

## Abstract

This paper presents a general formulation for optimization of horizontal road alignment, composed of tangential segments and circular curves suitably connected with transition curves (clothoids). It consists of a constrained optimization problem where the objective function is given by a line integral along the layout. The integrand is a function representing the cost of the road going through each point and, by considering different costs, a wide range of problems can be included in this formulation. To show it, we apply this methodology to three different situations. The two first cases are related with the design of a new road layout and used to solve a pair of academic examples. The third problem deals with the improvement of a road adapting the old path to current legislation, and it is solved taking as case study the reconstruction project for a regional road (NA-601) in the north of Spain.

Keywords: Road design, horizontal alignment, road alignment improvement, constrained optimization.

## 1. Introduction

The challenge of achieving the optimal alignment for a road design is a complex problem and highly topical in Civil Engineering. In general, the aim is to obtain an admissible layout minimizing the final cost of the working execution. The layout must also be admissible and fit the constraints given by the legislation of each country as well as the inherent characteristics of the location, for example, regions where the road must go through or restricted areas.

Mathematical modelling and optimization techniques can be a powerful tool seeking optimal alignment. Nowadays the bibliographic references are extensive in that topic. Papers can be classified according to different aspects. With respect to the target for optimizing, there are papers devoted to horizontal alignment only [13, 7, 17, 20, 22], to vertical alignment only [14, 9, 10] and to the three dimensional alignment [4, 14, 19, 15, 24, 11]. With respect to the cost, some papers work with a specific objective, as traffic safety [7], length [17] or earthwork [9, 10, 20], while others deal with costs including elements depending on the location (land acquisition, environmental cost, terrain, etc.), length (pavement, maintenance, ...), traffic safety (visibility, secure overtaking, etc.), and others [4, 14, 19, 15, 24, 11, 18]. These last papers also can be sub-classified from the formulation of the optimization problem into single-objective [4, 14, 15] and multi-objective optimization problems [19, 24, 11]. In any case, it should be noted that earthwork cost is one of the most important economic costs. Most of the papers dealing with alignment optimization work with earthwork cost and it is also studied in many other optimization papers [6, 8, 5, 3]. Finally, the optimization approach is also a very important task in this field and, for example, in [15], and more recently in [18], a classification of the papers according to this aspect can be found.

A horizontal road alignment should be a series of straights (tangents) and circular curves jointed by means of transition curves. Although at present time there exist different approaches [16], the best alternative for transition curves are clothoids whose equations are complex and hamper to obtain an explicit parametrization of the path. To avoid that problem, [4] select a polynomial type equation to adjust the road design. [17] work with a polygonal path, and only once the optimization process has ended, curves (clothoid-circular-clothoid) are built over the optimal polygonal route. In [13], [20] and [11] an alignment consisting of straights and circular curves are constructed in terms of their respective decision variables, and in [15] transition curves (clothoids) are incorporated to the horizontal alignment, but an explicit parametrization of the path is not given.

In this work we deal with the horizontal road design considering that the layouts are composed by tangents and circular curves properly connected with clothoids. The aim is to give a simple and general formulation of an optimization problem to obtain the horizontal road alignment. With this goal in mind, we begin (Section 2) by determining the decision variables of the problem and, taking as starting point a previous work [23], we detail an algorithm to compute a complete parametrization of the alignment from those variables. Afterwards, in Section 3, we introduce a standard approach to compute optimal layouts, based on framing the horizontal alignment design as a constrained optimization problem. The main idea lies in thinking that every geographical point has a price and that the objective function is the sum of all those prices creating the road. A wide interpretation of price allows us to include in this framework a large amount of diverse problems. To illustrate this fact we employ the proposed methodology to three different cases. In the first two (Section 4) we look for the shortest path joining two terminals, avoiding certain obstacles (protected areas to preserve or either zones to dodge due to high slopes). In both cases, we show the effectiveness of our formulation by solving the corresponding problems in simple academic examples. The third problem (Section 5) is related with a road reconstruction project (particularly, with improving alignment to adapt it to current legislation). In this case, we take as case study the road reconstruction project of a regional road (NA-601) in Navarra, northern Spain, and we compared our results with the real performed project. Finally (Section 6), some brief and interesting conclusions are sketched.

## 2. Mathematical model for horizontal alignment

### 2.1. Design variables

The problem arising is the design of a road between two given points \(a, b \in \mathbb{R}^2\). The horizontal road alignment to be implemented should be formed by the suitable combination of straight sections, circular curves and transition curves, which in our model will be clothoids. If the path consists of \(N + 1\) tangents, it is unequivocally determined by the vertices (\(v_i = (x_i, y_i), i = 1, \dots, N\)) where these tangents intersect, and the radii (\(R_i \ge 0, i = 1, \dots, N\)) and angles (\(\omega_i \ge 0, i = 1, \dots, N\)) of the circular curves, (see Fig. 1). Thus, for each \(N \in \mathbb{N}\) we define

\[
\mathbf{x}^N = (x_1, y_1, R_1, \omega_1, \dots, x_N, y_N, R_N, \omega_N) \in \mathbb{R}^{4N},
\]

the vector of the decision variables in the alignment optimization problem.
Moreover, we denote by \(C_{\mathbf{x}^N} \subset \mathbb{R}^2\) the curve (path of the road) determined by \(\mathbf{x}^N\).

**Remark 1.** If at one end \(a\) or \(b\) (or both) the road to be designed must connect with a given path, the decision variables must undergo small modifications:

1. If at point \(a\) (respectively \(b\)) the path joins with a circular curve of radius \(R_a \in (0, +\infty)\) (resp. \(R_b\)) with a given orientation, decision variables \(x_1\) and \(y_1\) (resp. \(x_N\) and \(y_N\)) must be replaced (see [23]) by \(\phi_1 \in [0, 2\pi]\) and \(d_1 \ge 0\) (resp. \(\phi_N\) and \(d_N\)) representing the azimuth of the first (resp. last) tangent and the distance from the connection point tangent-clothoid to the first (resp. last) vertex (see Fig. 2a).
2. If at point \(a\) (respectively \(b\)) the path connects with a straight section with azimuth \(\phi_a \in [0, 2\pi]\) (resp. \(\phi_b\)), the decision variables \(x_1\) and \(y_1\) (resp. \(x_N\) and \(y_N\)) should be replaced by \(d_1 \ge 0\) (resp. \(d_N\)) which represents the distance from point \(a\) (resp. \(b\)) to the first (resp. last) vertex (see Fig. 2b).

### 2.2. The arc length parametrization

Taking into account that \(C_{\mathbf{x}^N} \subset \mathbb{R}^2\) must be the union of straight segments and circular curves joined by clothoids, the road path can be easily parametrized in terms of the arc length parameter. We are assuming that the layout should begin and end with a straight section, thus we define \(v_0 = c_0 = a\), \(v_{N+1} = t_{N+1} = b\). Moreover, for \(i = 1, \dots, N\) and \(j = 1, \dots, N + 1\) we introduce the following functions and notation (see Fig. 3):

* **Unit vector giving the direction (and sense) of tangent \(j\):**
  \[
  \mathbf{u}_j(\mathbf{x}^N) = \frac{(x_j - x_{j-1}, y_j - y_{j-1})}{\sqrt{(x_j - x_{j-1})^2 + (y_j - y_{j-1})^2}}
  \]
* **Azimuth of tangent \(j\):**
  \[
  \phi_j(\mathbf{x}^N) = \begin{cases}
  \operatorname{acos}\left(\frac{y_j - y_{j-1}}{\sqrt{(x_j - x_{j-1})^2 + (y_j - y_{j-1})^2}}\right) & \text{if } x_j - x_{j-1} \ge 0 \\
  2\pi - \operatorname{acos}\left(\frac{y_j - y_{j-1}}{\sqrt{(x_j - x_{j-1})^2 + (y_j - y_{j-1})^2}}\right) & \text{if } x_j - x_{j-1} < 0
  \end{cases}
  \]
* **Difference of the azimuth between tangents \(i\) and \(i+1\):**
  \[
  \theta_i(\mathbf{x}^N) = |\phi_{i+1}(\mathbf{x}^N) - \phi_i(\mathbf{x}^N)|
  \]

* **Length of circular curve \(i\):**
  \[
  L_i^{CC}(\mathbf{x}^N) = R_i\omega_i. \tag{1}
  \]

* **Length of each clothoid in turn \(i\):**
  \[
  L_i^C(\mathbf{x}^N) = R_i(\theta_i(\mathbf{x}^N) - \omega_i). \tag{2}
  \]

* **Junction between the straight segment \(i\) and the beginning of turn \(i\):**
  \[
  t_i(\mathbf{x}^N) = v_i - (\overline{tp_i} + \overline{ph_i} + \overline{hv_i}) \mathbf{u}_i(\mathbf{x}^N).
  \]
  It can be clearly seen that (regarding [23]):
  \[
  \overline{tp_i} = \int_0^{L_i^C} \cos\left(\frac{\tau^2}{2R_i L_i^C}\right) d\tau, \quad \overline{p\hat{f_i}} = \int_0^{L_i^C} \sin\left(\frac{\tau^2}{2R_i L_i^C}\right) d\tau.
  \]
  \[
  \overline{ph_i} = \overline{pf_i} \tan\left(\frac{\theta_i - \omega_i}{2}\right), \quad \overline{hv_i} = \left(R_i + \frac{\overline{pf_i}}{\cos\left(\frac{\theta_i - \omega_i}{2}\right)}\right) \frac{\sin\left(\frac{\omega_i}{2}\right)}{\sin\left(\frac{\pi - \theta_i}{2}\right)}
  \]

* **Junction between the end of turn \(i\) and the beginning of the straight segment \(i+1\):**
  \[
  c_i(\mathbf{x}^N) = v_i + (\overline{tp_i}(\mathbf{x}^N) + \overline{ph_i}(\mathbf{x}^N) + \overline{hv_i}(\mathbf{x}^N)) \mathbf{u}_{i+1}(\mathbf{x}^N).
  \]

* **Length of straight segment \(j\) (_oriented_ distance from the end of turn \(j-1\) to the beginning of \(j\)):**
  \[
  L_j^T(\mathbf{x}^N) = r_j(\mathbf{x}^N) \cdot \mathbf{u}_j(\mathbf{x}^N),
  \]
  where \(r_j\) is the vector starting at point \(c_{j-1}\) and ending at point \(t_j\).

In order to \(C_{\mathbf{x}^N} \subset \mathbb{R}^2\) be a properly design of a horizontal road alignment, the following conditions must be satisfied:

1. The radii and angles of circular curves must be nonnegative, i.e., for \(i = 1, \dots, N\),
  \[
  R_i \ge 0, \\
  \omega_i \ge 0. \tag{3}
  \]
2. The angle of the circular curves must be lower or equal to the difference of azimuths between the corresponding tangents on every turn, i.e., 
  \[
    \theta_i(\mathbf{x}^N) - \omega_i \ge 0, \quad \forall i = 1, \dots, N. \tag{4}
  \]
3. Turn \(i + 1\) must begin after the end of turn \(i\), i.e.,
  \[
  L_{j}^T(\mathbf{x}^N) \ge 0, \quad \forall j = 1, \dots, N+1. \tag{5}
  \]

**Remark 2.** To guarantee that \(C_{\mathbf{x}^N} \subset \mathbb{R}^2\) represents a road alignment design one should think that it is also necessary to demand that lengths of circular curves (\(L_i^{CC}\)) and clothoids (\(L_i^C\)) must be nonnegative. However, taking into account (1) and (2), those constraints are automatically granted by (3) and (4).

If constraints (3)-(5) hold, the length of the road alignment prior to the beginning of turn \(j\), is given by,
\[
L_j(\mathbf{x}^N) = L_1^T(\mathbf{x}^N) + \sum_{k=1}^{j-1} 2L_k^C(\mathbf{x}^N) + L_k^{CC}(\mathbf{x}^N) + L_{k+1}^T(\mathbf{x}^N).
\]
Thus, the total length of the road alignment is given by \(L(\mathbf{x}^N) = L_{N+1}(\mathbf{x}^N)\), and the parametrization of curve \(C_{\mathbf{x}^N} \subset \mathbb{R}^2\) in terms of the arc length parameter is the mapping \(\sigma_{\mathbf{x}^N} \in C^2([0, L(\mathbf{x}^N)], \mathbb{R}^2)\) detailed in Algorithm 1 (it can be easily obtained with the method proposed in [23]).

**Algorithm 1: Computation of \(\sigma_{\mathbf{x}^N}(s)\)**
* _Initial straight segment:_ If \(s \in [0, L_1^T]\), compute
    \[
    \sigma(s) = a + s \mathbf{u}_1.
    \]

*   `For i = 1,...,N`
    *   _Ingoing clothoid:_ If \(s \in (L_i, L_i + L_i^C]\), compute
        \[
        \sigma(s) = t_i + \int_0^{\tilde{s}} \left(\cos\left(\frac{\lambda_i \tau^2}{2R_i L_i^C} + \beta_i\right) d\tau, \sin\left(\frac{\lambda_i \tau^2}{2R_i L_i^C} + \beta_i\right) d\tau\right)
        \]
        where
        *   \(\tilde{s} = s - L_i\),
        *   \(\beta_i \in [0, 2\pi)\) is the angle between \(\mathbf{u}_i\) and \(OX^+\) (see Fig. 3),
        *   \(\lambda_i \in \{-1, 1\}\) gives the orientation of the ingoing clothoid (the clothoid starting at \(t_i\)): \(\lambda_i = -1\) if it is traversed in the clockwise direction and \(\lambda_i = 1\) in counter-clockwise.
    *   _Circular curve:_ If \(s \in (L_i + L_i^C, L_i + L_i^C + L_i^{CC}]\), compute
        \[
        \sigma(s) = o_i + R_i\left(\cos\left(\alpha_i + \frac{\tilde{s}}{R_i}\right), \sin\left(\alpha_i + \frac{\tilde{s}}{R_i}\right)\right)
        \]
        where
        *   \(\tilde{s} = s - (L_i + L_i^C)\),
        *   \(h_i = v_i - \overline{hv_i} \mathbf{u}_i\),
        *   \(f_i = \sigma(L_i + L_i^C)\),
        *   \(\mathbf{w}_i = \frac{f_i - h_i}{\|f_i - h_i\|}\),
        *   \(o_i = f_i + R_i \mathbf{w}_i\),
        *   \(\alpha_i \in [0, 2\pi)\) the angle between \(\mathbf{w}_i\) and \(OX^+\) (see Fig. 3).
    *   **Outgoing clothoid:** If \(s \in (L_i + L_i^C + L_i^{CC}, L_i + 2L_i^C + L_i^{CC}]\), compute
        \[
        \sigma(s) = c_i + \int_0^{\tilde{s}} \left(\cos\left(\frac{\lambda_i \tau^2}{2R_i L_i^C} + \beta_i\right) d\tau, \sin\left(\frac{\lambda_i \tau^2}{2R_i L_i^C} + \beta_i\right) d\tau\right)
        \]
        with
        *   \(\tilde{s} = L_i + 2L_i^C + L_i^{CC} - s\).
    *   **Straight segment:** If \(s \in (L_{i+1} - L_{i+1}^T, L_{i+1}]\), compute
        \[
        \sigma(s) = c_i + \tilde{s}\mathbf{u}_{i+1},
        \]
        with
        *   \(\tilde{s} = s - (L_{i+1} - L_{i+1}^T)\).

## 3. Optimal design of horizontal alignment: a general formulation

3
Optimal design of horizontal alignment: a general formulation
Constraints (3)-(5) guarantee that CxN ⊂Ωis a well defined road alignment but, obviously, not all
paths have to be admissible. For instance, each country legislation usually has legal constraints over the
elements of the layout (bounds over the length of the clothoids and/or the straight segments, some kind
of relationship between radii of two consecutive curves, etc., see for example, [1]). Other usual constraints
are due to the existence of some regions into Ωwhere the route must not cross and others where going
through is prescribed. In a general way, all these restrictions, can be gathered in the set
Cad = {C ⊂R2, such that C is an admissible path for the new road}.
The set Cad depends on the particular problem we are dealing, and from it, we define the admissible set
in the following way:
XN
ad =
{
xN ∈R4N holding (3)-(5) and such that CxN ∈Cad
}
.
On the other hand, the aim pursued is to design a road alignment that, under certain technical aspects,
turns out to be optimal (minimizing length, earthwork, expropriation costs, environmental impact, ...).
The definition and the calculation of the objective function are crucial in any practical application. In the
aim of seeking for a simple and general formulation of the problem, we introduce a function F : Cad −→R
giving the cost (economic, environmental, ...) of every path. Therefore we define, JN : R4N −→R as
JN(xN) = F(CxN ),
and the problem of designing the optimal horizontal road alignment connecting a and b, consists of solving
the problem
min
xN∈XN
ad
JN(xN),
(6)
for each N = 1, 2, . . ., and choosing the CxN corresponding with the lowest value.
The mathematical function F (just like the set XN
ad) is characteristic for each particular problem, and
obtaining a good expression of F (bringing together all existing costs) can be a difficult task in many
practical applications. For example, in [20] earthwork cost is taken into account by means of a complete
vertical alignment subproblem. In this paper, to define function F we propose that every point of the
domain Ωhas a price (cost), so, in a way that there exists a function p(x, y) which gives the price of the
road passing through (x, y) ∈Ω. Adding the prices of all the locations which the road goes through, we
have the total cost of the layout given by
F(C) =
∫
C
p(x, y) dσ.
In this case, taking into account that function σxN (given by Algorithm 1) is the parametrization of CXN
in terms of the arc length parameter, the objective function JN is given by
JN(xN) =
∫L(xN)
0
p(σxN (s)) ds,
(7)
which can be evaluated by using a suitable numerical integration formula.
The concept of price, given by function p, should be understood as a general function that models a
wide range of possibilities: it can be economic (price of expropriation, asphalt cost, earthwork, etc.), but
also environmental, ecological or political. That price can also be seen as a penalization for going through
certain points, which allows to simplify the set Cad, by including the points where the layout must not cross into the objective function. Finally, p(x, y) can also be a combination (weighted sum) of different
types of prices.
If we want to take into account all existing costs, obtaining a function p, as occurs with function
F, can be a difficult task. However, in some simple applications, function p can be easily defined, as is
illustrated in next sections, with two practical cases. In Section 4 we look for the shortest layout avoiding
obstacles and/or minimizing slopes. In Section 5 we deal with improving the alignment of an existing
road, in order to adapt it to current legislation.

## 4. Design of a new road layout

We want to design the horizontal alignment for a new road connecting two terminals a, b ∈R, holding
the following restrictions: the radii of all circular curves should be at least 50 meters, straight segments
must be over 100 meters and the length of every clothoid should be, at least, 95 meters. In this situation,
the admissible set results
\[
X_{ad}^N = \left\{x^N \in \mathbb{R}^{4N} \left| \begin{array}{ll}
R_i \ge 0.05 \\
\theta_i(x^N) - \omega_i \ge 0.1 & i = 1, \dots, N \\
L_i^C(x^N) \ge 0.095 \\
L_j^T(x^N) \ge 0.1 & j = 1, \dots, N+1
\end{array} \right. \right\}. \tag{8}
\]
Regarding the objective function J, we study two different situations:

### 4.1. Minimizing length and avoiding obstacles

As first example we are interested in finding a road alignment of minimum length and avoiding trespassing zones \(A_1, \dots, A_{N_Z} \subset \Omega\). In this case we have to think that the price of every point of the domain is 1 if the point is outside the restricted area and a very high value (let us consider \(P_j >> 0\)) if the point is inside \(A_j\). This leads to the following price function:
\[
p(x, y) =
\begin{cases}
1 & \text{if } (x, y) \in \Omega \setminus \left(\cup_{j=1}^{N_Z} A_j\right), \\
P_j & \text{if } (x, y) \in A_j.
\end{cases}
\]
This discontinuous function \(p\) can be approached by a smooth approximation (\(\tilde{p}\)) in order to guarantee the smoothness of (7) and use differentiable optimization algorithms for solving problem (6). This methodology (and its good performance) is illustrated in the following academical example:
We seek for the shortest path starting from point a = (0, 1), reaching point b = (5.2, 2.1), fulfilling the restrictions defining the admissible set (8), and avoiding to cross through three circles centred in $c_1 = (1, 1)$, $c_2 = (2.3, 2.4)$, $c_3 = (4.2, 1.3)$ and radii $r_1 = 0.6$, $r_2 = 0.9$, $r_3 = 1$, respectively (see Fig 4).To do so, we define
\[
p_j(x, y) = r_j^2 - ||(x, y) - c_j||^2, \quad j = 1, 2, 3,
\]
\[
\tilde{p}(x, y) = 1 + 10^4 \sum_{j=1}^3 (max\{p_j(x, y), 0\})^2,
\]
and we solve the problem (6) for \(N = 1, 2, 3,\) using a Sequential Quadratic Programming (SQP) algorithm (see [21]), with a MATLAB code running on a Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz.
The complete process for the three cases took about 128 seconds CPU time. As expected, the best solution is reaching for \(N = 3\). The highlight of results can be seen in Table 1. The good behaviour of the method is shown in Fig. 4 where we plot the obstacles meant to be bypassed, and the optimal road alignment computed with one, two and three turns (\(N = 1\), \(N = 2\) and \(N = 3\)).

Table 1: Numerical results obtained for problem detailed in Section 4.1.

| N | \(v_i = (x_i, y_i)\) | \(R_i\) | \(\omega_i\) | \(L_i^C(x^N)\) | \(L_i^T(x^N)\) | \(L(x^N)\) | \(J(x^N)\) |
|---|----------------------|---------|-------------|---------------|---------------|-------------|------------|
| 1 | (2.034, 3.509)        | 0.901   | 0.954       | 0.319         | 2.377         | 6.485       | 6.485      |
|   |                      |         |             |               | 2.611         |             |            |
| 2 | (0.988, 0.253)        | 0.600   | 0.908       | 0.213         | 0.692         | 6.102       | 6.102      |
|   | (4.482, 2.724)        | 1.019   | 1.232       | 0.102         | 2.881         |             |            |
|   |                      |         |             |               | 0.100         |             |            |
| 3 | (0.902, 1.677)        | 0.388   | 0.205       | 0.308         | 0.757         | 5.660       | 5.660      |
|   | (2.084, 1.238)        | 1.520   | 0.777       | 0.152         | 0.101         |             |            |
|   | (4.040, 2.362)        | 0.968   | 0.644       | 0.097         | 1.041         |             |            |
|   |                      |         |             |               | 0.763         |             |            |

**Remark 3.** The SQP algorithm provides very good solutions in this case. However, it should be noted that the convexity of the objective function is not guaranteed in a general case and local minimizers can be detected. In that situation, global optimization methods should be considered. In the same way, other optimization algorithms, such as heuristic or metaheuristic methods, should be used in case of noisy and non-differentiable cost functions (see [17]).

### 4.2. Minimizing slope and length

In the second example we seek for the shortest horizontal alignment minimizing slopes. Let us assume that we have a function \(H(x, y)\) giving the height above sea level (\(km\)), of every point of \(\Omega\).

For a smooth parametrized curve \(C = \{\alpha(t) = (\alpha_1(t), \alpha_2(t)), t \in [t, \overline{t}]\}\), the slope of each point \(\alpha(t) \in C\) (using the chain rule) is given by
\[
H'(t) = \frac{\partial H}{\partial x} (\alpha(t)) \alpha'_1(t) + \frac{\partial H}{\partial y} (\alpha(t)) \alpha'_2(t).
\]
This leads to choose the following objective function
\[
J_N(\mathbf{x}^N) = \int_0^{L(\mathbf{x}^N)} \left(\epsilon + (1 - \epsilon) \left(\frac{\partial H}{\partial x} (\sigma_{\mathbf{x}^N}(s)) \sigma'_{\mathbf{x}^N, 1}(s) + \frac{\partial H}{\partial y} (\sigma_{\mathbf{x}^N}(s)) \sigma'_{\mathbf{x}^N, 2}(s)\right)^2 \right) ds,
\]
where
- \(\epsilon \in [0, 1]\) is a weight parameter that quantifies the relevance of length of the layout versus its slope (if the value of \(\epsilon\) is close to zero, minimizing the slope is the main objective, and if it is close to one, the main objective lies to obtain a short layout).
- Function \(\sigma_{\mathbf{x}^N}(s) = (\sigma_1(s), \sigma_2(s))\) is for each case, the parametrization of \(C_{\mathbf{x}^N}\) given in Algorithm 1.

To show the good performance of this formulation we have chosen the following academical example: as physical domain we have \(\Omega = [-1, 6] \times [-2, 6]\) and we seek for a road with terminals \(a = (0.25, 1)\) and \(b = (4, 2)\). The orography of the domain, given by function \(H(x, y)\) detailed in Appendix 1, is shown in Fig. 5. We have chosen \(\epsilon = 10^{-3}\) as weight parameter between length and slope. Keeping in mind \(X_{ad}^N\) and \(J_N\) given in (8) and (9), respectively, we solve problem (6), for \(N = 1, 2, 3\), using again a SQP method. The best solution is obtained for \(N = 3\) and the numerical results are shown in Table 2. The optimal road alignments are shown in Fig. 6. It exemplifies how the horizontal road alignments cross the contour lines. The goodness of obtained solutions can be also regarded in Fig. 7, where the layout of those optimal solutions are plotted in a 3D vision. In this example, the computation time for the three alignments took about 854 seconds CPU time.
If only one turn is allowed (\(N = 1\)), the optimal solution lies in circumventing the mountains doing a substantial detour adjusting the route to go roughly at the same high level. This leads to a huge circular curve which can be not desirable from a “road design” point of view. However shows the good behaviour of the proposed method (in this situation, to use this method from a more realistic point of view, the constraints defining the admissible set \(X_{ad}^N\) must be reviewed, including, for instance, constraints over maximum lengths of circular curves).

If more than one turn is permitted, the optimal solution turns out a layout going through the mountains, heavily decreasing the length of the path. It also can be seen that in both cases (\(N = 2\) and \(N = 3\)) the layout tries to avoid steep slopes, obtaining, as it might be expected, that the alignment with 3 turns (\(N = 3\)) is the best solution.

Table 2: Numerical results obtained for problem detailed in Section 4.2.

|  | \(v_i = (x_i, y_i)\) | \(R_i\) | \(\omega_i\) | \(L_i^C\) | \(L_i^T\) | \(L\) | \(J\) |
|---|----------------------|---------|-------------|---------------|---------------|-------------|------------|
| \(N=1\) | (2.86, 14.99)         | 1.624   | 2.190       | 1.100         | 1.593         | 7.755       | 0.0095     |
|   |                      |         |             |               | 2.611         |             |            |
| \(N=2\) | (0.96, 0.52)          | 0.404   | 0.160       | 0.423         | 0.355         | 4.409       | 0.0084     |
|   | (3.51, 2.29)          | 0.210   | 0.431       | 0.150         | 2.395         |             |            |
|   |                      |         |             |               | 0.361         |             |            |
| \(N=3\) | (0.87, 2.72)          | 0.349   | 0.094       | 0.636         | 0.950         | 4.943       | 0.0076     |
|   | (2.33, 1.51)          | 0.687   | 0.046       | 0.736         | 0.205         |             |            |
|   | (3.55, 2.06)          | 0.225   | 0.138       | 0.095         | 0.412         |             |            |
|   |                      |         |             |               | 0.346         |             |            |

If more than one turn is permitted, the optimal solution turns out a layout going through the mountains, heavily decreasing the length of the path. It also can be seen that in both cases (\(N = 2\) and \(N = 3\)) the layout tries to avoid steep slopes, obtaining, as it might be expected, that the alignment with 3 turns (\(N = 3\)) is the best solution.

## 5. Improvement of horizontal alignment

# 5 Improvement of horizontal alignment

In this section we use the general formulation proposed in Section 3 to improve the alignment of an existing road. We assume that the old road needs an upgrade and the layout needs to be adapted to current legislation (for instance, the enforceability of using transition curves, restrictions over radii, length of straight sections, etc.). The aim here is the design of a new horizontal alignment linking points \( a \) and \( b \). This should be done exploiting as much as possible the old layout and verifying the legislation constraints.

For this purpose, we suppose that the old layout is the graph of a known function
\[
y_{old}:[x_{in},x_{end}] \subset \mathbb{R} \longrightarrow \mathbb{R},
\]
where, of course, \( a=(x_{in},y_{old}(x_{in})) \) and \( b=(x_{end},y_{old}(x_{end})) \).

Obviously, the first step consists in condensing in set \( C_{ad} \) all constraints that the new layout must fulfill, and afterward determine \( X_{ad}^N \). This must be done for every particular problem and, in the following section, we specify this set for a particular case-study.

The second step is to define an objective function providing the quality of every path. If our goal is to exploit as much as possible the existing road, this can be done setting a price for every point by means of the \( y \)-distance to the old road: zero price for all points which are on the old road (intersect the old road), price one for all those points whose \( y \)-distance exceeds a certain maximum threshold \( d_{max} \) (over that value the old road is no longer exploitable) and, for the other, the price is a monotonous increasing function of the \( y \)-distance. Furthermore, in order to use a derivative-based optimization method to solve the problem (6), we propose to take
$$
p(x,y) = p_{y_{old}(x)}(y), \tag{10}
$$
where, for every \( y_0 \in \mathbb{R} \), the function \( p_{y_0} \) is given by (see Fig. 8(a)):

$$
p_{y_0}(y) = \begin{cases}
    \frac{2}{d_{max}^2}(y-y_0)^2 - \frac{1}{d_{max}^4}(y-y_0)^4 & \text{if } y_0 - d_{max} \le y < y_0 + d_{max}, \\
    1 & \text{other case.}
\end{cases} \tag{11}
$$

The price function \( p(x,y) \) is like a canyon along the old road. If the new layout abandons the old path (if \( y \)-distance is greater than \( d_{max} \)), a new motorway must be constructed and we assume equal cost for any new built section (we neglect other costs and we only take care to minimize the length of the horizontal alignment).

We will now apply this methodology to a particular problem.

### 5.1. Case study: NA601 (Campanas-Lerín, Spain) Road Reconstruction Project

To test the method we are proposing, we have chosen a section of the road joining Larraga-Lerín (NA601) in Navarra, Spain. This road has been already improved following a road reconstruction project approved by the regional government in December, 3rd 2001 (the completion of construction work ended in March, 2nd 2007). The old road layout and the new one (current) can be seen in Fig. 9.
In order to use our methodology to this particular case, we perform the following steps:
- Step 1: Due to the constraints that the new layout must verify, we define the admissible set \(X_{ad}^N\).
In this paper we assume that the radii of all circular curves must be at least 30 meters, the length of circular curves must be between 80 and 450 meters, the length of every clothoid should be between 85 and 1300 meters, and straight segments must be between 100 and 2500 meters. Derived from this constraints we define
    \[
    X_{ad}^N = \left\{x^N \in \mathbb{R}^{4N} \left| \begin{array}{ll}
    R_i \ge 0.3 \\
    \theta_i(x^N) - \omega_i \ge 0.1 \\
    0.080 < L_i^C(x^N) \le 0.450 \\
    0.085 < L_i^{CC}(x^N) \le 1.300 \\
    0.1 \le L_j^T(x^N) \le 2.500
    \end{array} \right. \begin{array}{l}
    i = 1, \dots, N \\
    j = 1, \dots, N+1
    \end{array} \right\} \tag{12}
    \]
*   **Step 2:** From the coordinates of some suitable points of the old layout we construct the \(y_{old}\)
function by using a cubic spline interpolation (see [2]).
*   **Step 3:** Taking \(d_{max} = 0.07\) km, from \(Y_{old}\) built in Step 2, we define function \(p(x, y)\) given by (10), (see Fig. 8(b)), and \(J^N\) by (7).
*   **Step 4:** Given the admissible set \(X_{ad}^N\) (Step 1) and function \(J^N\) (Step 3), for each \(N = 1, 2, \dots\), we solve the problem (6) by using again a SQP method.

The complete processing time for the first four problems (\(N \leq 4\)) took about 37 seconds CPU time. As Table 3 shows, the best option for the road improvement is the one with four turns (\(N = 4\)). The optimal layouts computed are plotted in Fig. 10. It shows that, as the number of turns is increased, a better adjustment to the old layout is obtained. For \(N = 1\) as only one turn is allowed, the optimal solution lies in intersect the longest straight sector of the old path. Finally, in Fig. 11 we compare the current road improvement versus our proposal obtained with \(N = 3\). It shows that both are similar. This is a crucial and relevant remark and make us believe that the methodology proposed can be a good way to deal with improving alignments in road reconstruction projects as the one introduced in this section.

Table 3: Numerical results obtained for problem detailed in Section 5.1. Vertices \(v_i\) are given in ERTS89 coordinates (km).

| N | \(v_i = (x_i, y_i)\) | \(R_i\) | \(\omega_i\) | \(L_i^C\) | \(L_i^T\) | \(L\) | \(J\) |
|---|----------------------|---------|-------------|-----------|-----------|-------|-------|
| 1 | (587.446, 4706.035)   | 0.300   | 0.136       | 0.080     | 0.721     | 3.187 | 2.028 |
|   |                      |         |             |           | 2.266     |       |       |
| 2 | (587.540, 4706.097)   | 0.300   | 0.853       | 0.080     | 0.758     | 3.316 | 0.456 |
|   | (588.806, 4705.742)   | 1.729   | 0.786       | 0.192     | 0.211     |       |       |
|   |                      |         |             |           | 0.269     |       |       |
| 3 | (587.533, 4706.093)   | 0.300   | 0.558       | 0.080     | 0.755     | 3.311 | 0.338 |
|   | (588.381, 4705.879)   | 0.494   | 0.326       | 0.080     | 0.540     |       |       |
|   | (589.308, 4706.106)   | 0.300   | 0.115       | 0.080     | 0.694     |       |       |
|   |                      |         |             |           | 0.479     |       |       |
| 4 | (587.527, 4706.090)   | 0.300   | 0.544       | 0.080     | 0.751     | 3.343 | 0.050 |
|   | (588.503, 4705.858)   | 0.302   | 0.538       | 0.080     | 0.665     |       |       |
|   | (588.846, 4706.078)   | 0.300   | 0.374       | 0.080     | 0.100     |       |       |
|   | (589.234, 4706.051)   | 0.300   | 0.428       | 0.080     | 0.100     |       |       |
|   |                      |         |             |           | 0.521     |       |       |

## 6. Conclusions

The design of horizontal road alignment composed of a sequence of tangents, circular curves and clothoids is uniquely established by the intersection of the tangents (vertices), and the radii and angles of the circular curves. In this paper we develop an algorithm capable of obtaining, from those values, a complete parametrization of the layout in terms of the arc length parameter. The wide interpretation of price enables to tackle diverse problems as shown in the examples included in Sections 4 and 5. The obtained results endorse the suggested method as a good tool to seek the initial alignment for a new road. Moreover, the results obtained in Section 5.1 lead us to believe that this methodology can be a good way to deal with improving alignments in road reconstruction projects as the one introduced in that section. Finally, it is convenient to remark that this formulation can be also developed in layouts with vertical alignment. This extension will be studied in a further work to tackle more complex and interesting problems, such as the design with minimum earthwork. More complete objective functions will be more difficult to optimize and SQP may no longer work, which will require the use of other suitable optimization methods.

### Acknowledgements
The authors gratefully thank María José Enríquez-García (LaboraTe, USC), for her help at the field of GIS. Corresponding author also thanks the support given by Project MTM2015-65570-P of MEC (Spain) and FEDER.

### Appendix A: Analytic description of height \(H(x, y)\) for academic example in section 4.2

In order to give and analytic description of function \(H(x, y)\) used in the academic example of section 4.2, we consider, for \(i = 1, \dots, 5\), the values of parameters \(s_i = (s_i^1, s_i^2) \in \mathbb{R}^2\), \(l_i \in \mathbb{R}\), \(m_i \in \mathbb{R}\), \(n_i \in \mathbb{R}\), \(d_i \in \mathbb{R}\) and \(q_{min}^i \in \mathbb{R}\) given in table 4. From these parameters we define the functions
\[
q_i(x, y) = d_i \left(l_i^2 - \left(\frac{x - s_i^1}{m_i}\right)^2 + \left(\frac{y - s_i^2}{n_i}\right)^2 \right) 
\]

Table 4: Values of the parameters used in the definition of function \(H(x,y)\).

|  | \(s_i = (s_i^1, s_i^2)\) | \(l_i\) | \(m_i\) | \(n_i\) | \(d_i\) | \(q_{min}^i\) |
|-----|------------------------|---------|---------|---------|---------|---------------|
| \(i=1\)   | (1, 1.5)               | 1       | 0.5     | 1       | 0.3     | 0             |
| \(i=2\)   | (2.5, 3.1)             | 1       | 1       | 1.5     | 0.4     | 0             |
| \(i=3\)   | (3.8, 1.3)             | 1       | 2       | 0.5     | 0.3     | 0             |
| \(i=4\)   | (1.75, -0.25)          | 0.75    | 1       | 1       | 0.5     | 0             |
| \(i=5\)   | (2.5, 2)               | 1       | 10      | 10      | 0.8     | 0.7           |

and, finally, we compute
\[
H(x, y) = 1 + \sum_{i=1}^5 (\max\{q_i(x, y), q_i^{min}\})^2
\]

### Appendix B: List of symbols and notation

| Symbol | Description |
|---|---|
| \(a = (x_a, y_a)\) | Initial point of the road with coordinates \((x_a, y_a)\). |
| \(b = (x_b, y_b)\) | Final point of the road with coordinates \((x_b, y_b)\). |
| \(N\) | Number of curves of the road joining \(a\) and \(b\). |
| \(v_i = (x_i, y_i)\) | Vertex defined by the intersection of tangents \(i\) and \(i+1\), with coordinates \((x_i, y_i)\). |
| \(R_i\) | Radius of the circular curve \(i\). |
| \(\omega_i\) | Angle of the circular curve \(i\). |
| \(\mathbf{x}^N\) | Vector of the decision variables. |
| \(C_{\mathbf{x}^N}\) | Path determined by \(\mathbf{x}^N\). |
| \(d_1, d_n\) | Distance from the connection point tangent-clothoid to the first (resp. last) vertex. |
| \(\mathbf{u}_j\) | Unit vector giving the direction (and sense) of tangent \(j\). |
| \(\phi_j\) | Azimuth of tangent \(j\). |
| \(\theta_i\) | Difference of the azimuth between tangents \(i\) and \(i+1\). |
| \(L_i^{CC}\) | Length of circular curve \(i\). |
| \(L_i^C\) | Length of each clothoid in turn \(i\). |
| \(L_j^T\) | Length of straight segment \(j\). |
| \(L_j\) | Length of the road alignment prior to the beginning of turn \(j\). |
| \(L\) | Length of the road alignment. |
| \(t_i\) | Junction between the straight segment \(i\) and the beginning of turn \(i\). |
| \(c_i\) | Junction between the end of turn \(i\) and the beginning of the straight segment \(i+1\). |
| \(r_j\) | Vector starting at point \(c_{j-1}\) and ending at point \(t_j\). |
| \(o_i\) | Center of the circular curve \(i\). |
| \(f_i\) | Junction between the ingoing clothoid and circular curve in turn \(i\). |
| \(p_i\) | Intersection between tangent \(i\) and its orthogonal line through \(f_i\). |
| \(h_i\) | Intersection between tangent \(i\) and the straight line defined by \(o_i\) and \(f_i\). |
| \(\overline{tp}_i\) | Distance between \(t_i\) and \(p_i\). |
| \(\overline{pf}_i\) | Distance between \(p_i\) and \(f_i\). |
| \(\overline{ph}_i\) | Distance between \(p_i\) and \(h_i\). |
| \(\overline{hv}_i\) | Distance between \(h_i\) and \(v_i\). |
| \(\sigma_{\mathbf{x}^N} = (\sigma_1, \sigma_2)\) | Parametrization of curve \(C_{\mathbf{x}^N}\) in terms of the arc length parameter. |
| \(s\) | Arc length parameter. |
| \(\beta_i\) | Angle between \(\mathbf{u}_i\) and \(OX^+\). |
| \(\lambda_i \in \{-1, 1\}\) | Parameter giving the orientation of the ingoing clothoid in turn \(i\). |
| \(\mathbf{w}_i\) | Vector from \(f_i\) to \(o_i\). |
| \(\alpha_i\) | Angle between \(\mathbf{w}_i\) and \(OX^+\). |
| \(Cad\) | Set of admissible paths for a new road. |
| \(X_{ad}^N\) | Admissible set (set of admissible decision variables vectors). |
| \(\Omega\) | Domain where the paths are defined. |
| \(F\) | Function giving the cost of every path. |
| \(J^N\) | Objective function of the optimization problem for \(N\) turns. |
| \(p\) | Function giving the price of the road passing through each point of \(\Omega\). |
| \(\tilde{p}\) | Smooth approximation of \(p\). |
| \(A_j\) | Trespassing area to be avoided. |
| \(P_j\) | Value of the price function if a path's point is inside \(A_j\). |
| \(H\) | Function giving the height above sea level. |
| \(\epsilon\) | Weight parameter. |
| \(Y_{old}\) | Function giving the old layout. |
| \(d_{max}\) | Value from which the old road is no longer exploitable. |
| \(p_{y_0}\) | Auxiliar function. |

### References

[1] AASHTO (2011). *A Policy on Geometric Design of Highways and Streets, 6th Edition.* American Association of State Highway and Transportation Officials. Washington D.C.

[2] Boor, C. (1978). *A Practical Guide to Splines.* Springer Series in Applied Mathematics Sciences, Springer-Verlag, New York.

[3] Burdett, R.L. and Kozan, E. (2014) “An integrated approach for earthwork allocation, sequencing and routing.” *Eur. J. Oper. Res.*, 238 (3), 741-759.

[4] Chew, E.P., Goh, C.J. and Fwa, T.F. (1989) “Simultaneous optimization of horizontal and vertical alignments for highways.” *Transp. Res. Pt. B-Methodol.*, 23 (5), 315-329.

[5] de Lima, R., Júnior, E., Prata, B. and Weissmann, J. (2013) “Distribution of Materials in Road Earthmoving and Paving: Mathematical Programming Approach.” *J. Constr. Eng. Manage.*, 139 (8), 1046-1054.

[6] Easa, S. M. (1988) “Earthwork allocations with linear unit costs.” *J. Constr. Eng. Manage.*, 114, 641-655.

[7] Easa, S. M. and Mehmood, A. (2008) “Optimizing Design of Highway Horizontal Alignments: New Substantive Safety Approach.” *Comput.-Aided Civil Infrastruct. Eng.*, 23, 560-573.

[8] Hare, W., Koch, V.R. and Lucet, Y. (2011) “Models and algorithms to improve earth work operations in road design using mixed integer linear programming.” *Eur. J. Oper. Res.*, 215(2), 470-480.

[9] Hare, W., Hossain, S., Lucet, Y. and Rahman, F. (2014) “Models and strategies for efficiently determining an optimal vertical alignment of roads.” *Comput. Oper. Res.*, 44, 161-173.

[10] Hare, W., Lucet, Y. and Rahman, F. (2015) “A mixed-integer linear programming model to optimize the vertical alignment considering blocks and side-slopes in road construction.” *Eur. J. Oper. Res.*, 241(3), 631-641.

[11] Hirpa, D., Hare, W., Lucet, Y., Pushak, Y. and Tesfamariam, S. (2016) “A bi-objective optimization framework for three-dimensional road alignment design.” *Transp. Res. Pt. C-Emerg. Technol.*, 65, 61-78.

[12] Jong, J.-C., Jha, M.K. and Schonfeld, P. (2000) “Preliminary highway design with genetic algorithms and geographic information systems” *Comput.-Aided Civil Infrastruct. Eng.*, 15, 261-271.

[13] Jong, J.-C. and Schonfeld, P. (2003) “Applicability of highway alignment optimization models” *Transp. Res. Pt. C-Methodol.*, 37 (2), 107-128.

[14] Kang, M.-W., Jha, M.K. and Schonfeld, P. (2012) “Applicability of highway alignment optimization models” *Transp. Res. Pt. C-Emerg. Technol.*, 21 (1), 257-286.

[15] Kobryń, A. (2014). “New Solutions for General Transition Curves.” *J. Surv. Eng.-ASCE*, 140 (1), 12-21.

[16] Lee, Y., Tsou, Y. and Liu, H. (2009) “Optimization Method for Highway Horizontal Alignment Design.” *J. Transp. Eng.*, 4, 217-224.

[17] Li, W., Pu, H., Schonfeld, P., Zhang, H. and Zheng, X. (2016) “Methodology for optimizing constrained 3-dimensional railway alignments in mountainous terrain” *Transp. Res. Pt. C-Methodol.*, 68, 549-565.

[18] Maji, A. and Jha, M.K. (2009) “Multi-Objective highway alignment optimization using genetic algorithm” *J. Adv. Transp.*, 43 (4), 481-504.

[19] Mondal, S., Lucet, Y. and Hare, W. (2015). “Optimizing horizontal alignment of roads in a specified corridor” *Comput. Oper. Res.*, 64, 130-138.

[20] Nocedal, J. and Wright, S.J. (2006). *Numerical Optimization.* Springer Series in Operations Research and Financial Engineering, Springer Science+Business Media, New York.

[21] Pushak, Y., Hare, W. and Lucet, Y. (2016). “Multiple-path selection for new highway alignments using discrete algorithms” *Eur. J. Oper. Res.*, 248, 415-427.

[22] Vázquez-Méndez, M.E. and Casal, G. (2016). “The clothoid computation: a simple and efficient numerical algorithm.” *J. Surv. Eng.-ASCE*, 142 (3), 04016005:1-9.

[23] Yang, N., Kang, M.-W., Schonfeld, P. and Jha, M.K. (2014) “Multi-objective highway alignment optimization incorporating preference information” *Transp. Res. Pt. C-Emerg. Technol.*, 40, 36-48.