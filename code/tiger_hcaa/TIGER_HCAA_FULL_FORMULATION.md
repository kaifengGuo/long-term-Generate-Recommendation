# TIGER-HCAA 完整数学方案

## 1. 问题设定

我们考虑一个 session 由多页推荐组成。记第 \(t\) 页展示之前的用户边界状态为
\[
h_{t-1},
\]
展示第 \(t\) 页并观察用户反馈之后的边界状态为
\[
h_t,
\]
其中 \(t=1,\dots,T\)。

第 \(t\) 页推荐的是一个 slate:
\[
a_t=\{i_{t,1},i_{t,2},\dots,i_{t,K_t}\},
\]
其中 \(K_t\) 是该页 item 数量。

每个 item \(i_{t,k}\) 由一个 token 序列表示:
\[
\tau_{t,k}=(\tau_{t,k,1},\tau_{t,k,2},\dots,\tau_{t,k,L_{t,k}}),
\]
其中 \(L_{t,k}\) 是 item \(k\) 的 token 长度。

我们希望构建一个三层长期价值归因体系:

1. `page` 层: 第 \(t\) 页整体对长期价值变化贡献了多少。
2. `item` 层: 第 \(t\) 页里的各个 item 如何分解这一页的长期贡献。
3. `token` 层: 每个 item 内部哪些 token 继续分解该 item 的贡献。

目标是得到
\[
A_t^{\mathrm{page}}
\;\Longrightarrow\;
A_{t,k}^{\mathrm{item}}
\;\Longrightarrow\;
A_{t,k,j}^{\mathrm{tok}}
\]
并满足层间守恒:
\[
\sum_{k=1}^{K_t} A_{t,k}^{\mathrm{item}} = A_t^{\mathrm{page}},
\qquad
\sum_{j=1}^{L_{t,k}} A_{t,k,j}^{\mathrm{tok}} = A_{t,k}^{\mathrm{item}}.
\]

---

## 2. 长期奖励与 session 势函数

### 2.1 长期奖励

记第 \(t\) 页的即时奖励为 \(r_t\)，退出或流失指示为 \(\xi_t \in \{0,1\}\)。定义长期奖励:
\[
r_t^{\mathrm{LT}} = r_t - \lambda_h \xi_t,
\]
其中 \(\lambda_h \ge 0\) 是退出惩罚系数。

它表达的是: 这一页不仅看即时反馈，还要扣掉“把用户推向退出”的代价。

### 2.2 session 势函数

定义从边界状态 \(h_t\) 出发的未来长期价值为
\[
Q^{\mathrm{sess}}(h_t)
=
\mathbb{E}
\left[
\sum_{m=t+1}^{T}
\gamma^{\,m-(t+1)} r_m^{\mathrm{LT}}
\;\middle|\;
h_t
\right].
\]

这里 \(Q^{\mathrm{sess}}(h_t)\) 的含义是:

- 用户已经走到了边界状态 \(h_t\)
- 从下一页开始，未来还能贡献多少长期价值

因此 \(Q^{\mathrm{sess}}(h_t)\) 是一个 session 级别的“势函数”或“剩余长期价值”。

### 2.3 page 级 action value

若第 \(t\) 页展示动作是 \(a_t\)，则其长期 action value 可写为
\[
Q^{\mathrm{page}}(h_{t-1}, a_t)
=
r_t^{\mathrm{LT}} + \gamma Q^{\mathrm{sess}}(h_t).
\]

这个量表示:

- 当前页带来的长期即时奖励 \(r_t^{\mathrm{LT}}\)
- 加上它把用户带到新状态后产生的未来长期价值 \(\gamma Q^{\mathrm{sess}}(h_t)\)

---

## 3. Page 粒度: 长期价值差分

### 3.1 Monte Carlo 长期回报

为了给 page 层提供硬监督，定义第 \(t\) 页起的 Monte Carlo 长期回报:
\[
G_t
=
\sum_{m=t}^{T} \gamma^{\,m-t} r_m^{\mathrm{LT}},
\qquad
G_{T+1}=0.
\]

于是有
\[
Q^{\mathrm{sess}}(h_{t-1}) \approx G_t,
\qquad
Q^{\mathrm{sess}}(h_t) \approx G_{t+1}.
\]

### 3.2 page advantage

我们把第 \(t\) 页的长期贡献定义成边界势函数的差分，再减去 page bias:
\[
A_t^{\mathrm{page}}

=
Q^{\mathrm{sess}}(h_t)
-
Q^{\mathrm{sess}}(h_{t-1})
-
b_t.
\]

这里 \(b_t\) 是 page 位置偏置，用于消除系统性的页位趋势。一个自然做法是先学一个原始 bias \(\bar b(p_t)\)，再做 session 内中心化:
\[
b_t
=
\bar b(p_t)
-
\frac{1}{T}\sum_{s=1}^{T}\bar b(p_s),
\]
于是自动满足
\[
\sum_{t=1}^{T} b_t = 0.
\]

这样做的好处是 page advantage 不再被“第几页天生就更容易高/低”这种位置效应污染。

### 3.3 page hard target

由 \(Q^{\mathrm{sess}}(h_{t-1}) \approx G_t\) 和 \(Q^{\mathrm{sess}}(h_t) \approx G_{t+1}\)，可得到 page 的硬目标:
\[
Y_t^{\mathrm{page}}
=
G_{t+1} - G_t - b_t.
\]

这正对应你希望的那种形式:
\[
\text{后一个时间步的 } Q - \text{前一个时间步的 } Q - \text{bias}.
\]

也就是说，page 粒度关注的是:

- 这页之后，用户长期状态有没有变好
- 变化了多少
- 去掉页位偏置后，剩下的纯长期贡献是多少

### 3.4 page 预测器

令模型预测
\[
\hat Q^{\mathrm{sess}}(h_{t-1}),
\qquad
\hat Q^{\mathrm{sess}}(h_t),
\qquad
\hat b_t,
\]
则 page advantage 的预测为
\[
\hat A_t^{\mathrm{page}}
=
\hat Q^{\mathrm{sess}}(h_t)
-
\hat Q^{\mathrm{sess}}(h_{t-1})
-
\hat b_t.
\]

### 3.5 page 层损失

page 层可以拆成四部分:
\[
L_{\mathrm{pre}}
=
\sum_t
\ell\!\left(
\hat Q^{\mathrm{sess}}(h_{t-1}),
G_t
\right),
\]
\[
L_{\mathrm{post}}
=
\sum_t
\ell\!\left(
\hat Q^{\mathrm{sess}}(h_t),
G_{t+1}
\right),
\]
\[
L_{\mathrm{bias}}
=
\sum_t
\ell(\hat b_t, b_t),
\]
\[
L_{\mathrm{page}}
=
\sum_t
\ell\!\left(
\hat A_t^{\mathrm{page}},
Y_t^{\mathrm{page}}
\right),
\]
其中 \(\ell\) 可以取 Huber loss 或 MSE。

---

## 4. 为什么 item 不能再用手工 target

旧链路中常见的 item target 往往直接用下面几类信号拼出来:

1. item reward
2. response 强度
3. support/prefix 匹配
4. 各种手工加权规则

这种做法的问题是:

1. 它不是长期价值模型自己推出的归因，而是人为规则。
2. 它回答不了“到底哪个 item 改变了长期价值”。
3. 它和 page 层的长期价值目标没有严格耦合。
4. 它不利于形成真正有学术味的分层长期价值解释。

因此，新方案里:

- `page` 层保留硬长期 target
- `item` 和 `token` 层不再手工设 target
- 而是通过 teacher critic 的可微归因自动生成 pseudo-target

---

## 5. Item 粒度: 对 page advantage 做可微分账

### 5.1 item 表示而不是 item id

对于第 \(t\) 页第 \(k\) 个 item，不直接对离散 item id 做归因，而是对它在当前上下文下的连续表示做归因:
\[
z_{t,k}
=
f_{\mathrm{item}}(h_{t-1}, x_{t,k}, a_t),
\]
其中:

- \(x_{t,k}\) 是 item 的原始信息，例如 item id、内容特征、位置特征、SID/token 等
- \(h_{t-1}\) 是当前用户状态
- \(a_t\) 表示当前页上下文
- \(z_{t,k}\) 是“这个 item 在这个用户、这个 slate、这个时刻”的语义表示

之所以不用裸 item id，有两个原因:

1. `item id` 是离散变量，不能做稳定的梯度归因。
2. 同一个 item 在不同用户状态、不同页位置、不同 slate 环境下贡献可以完全不同。

所以 item 层真正应该解释的不是“这个 id 好不好”，而是
\[
z_{t,k}
\]
这个上下文化连续表示对 page 长期价值差分的边际贡献。

### 5.2 teacher page critic

引入一个冻结或 EMA 更新的 teacher page critic:
\[
\mathcal{T}^{\mathrm{page}}_{\bar\theta}
\left(
h_{t-1}, z_{t,1:K_t}
\right)
\approx
Y_t^{\mathrm{page}}.
\]

这里 \(\bar\theta\) 是 teacher 参数。它的作用不是直接作为最终预测器，而是给 item 分账提供一个“长期价值导向的可微老师”。

### 5.3 item attribution score

记空白 item 的基准表示为 \(z_{\varnothing}\)。对 item \(k\) 的 attribution score 定义为 teacher page critic 对 \(z_{t,k}\) 的积分梯度:
\[
s_{t,k}^{\mathrm{item}}
=
(z_{t,k}-z_{\varnothing})^\top
\int_{0}^{1}
\nabla_{z_{t,k}}
\mathcal{T}^{\mathrm{page}}_{\bar\theta}
\left(
h_{t-1},
z_{t,1},
\dots,
z_{\varnothing}+\alpha(z_{t,k}-z_{\varnothing}),
\dots,
z_{t,K_t}
\right)
d\alpha.
\]

直觉上，这个量表示:

- 把第 \(k\) 个 item 从“空白”逐渐插入到当前页
- 观察 teacher page critic 的输出怎么变化
- 累计这个变化，就得到该 item 对 page 长期贡献的边际影响

若考虑效率，可用一阶近似:
\[
s_{t,k}^{\mathrm{item}}
\approx
\left\langle
\nabla_{z_{t,k}}
\mathcal{T}^{\mathrm{page}}_{\bar\theta},
\;
z_{t,k}-z_{\varnothing}
\right\rangle.
\]

### 5.4 sign-aware item share target

若该页长期贡献为正，即
\[
Y_t^{\mathrm{page}} \ge 0,
\]
则重点找“哪些 item 在正向拉高长期价值”。定义
\[
\tilde s_{t,k}^{\mathrm{item}} = [s_{t,k}^{\mathrm{item}}]_+.
\]

若该页长期贡献为负，即
\[
Y_t^{\mathrm{page}} < 0,
\]
则重点找“哪些 item 在拉低长期价值”。定义
\[
\tilde s_{t,k}^{\mathrm{item}} = [-s_{t,k}^{\mathrm{item}}]_+.
\]

然后做归一化:
\[
\alpha_{t,k}^{*}
=
\frac{
\tilde s_{t,k}^{\mathrm{item}}+\varepsilon
}{
\sum_{m=1}^{K_t}
\left(
\tilde s_{t,m}^{\mathrm{item}}+\varepsilon
\right)
},
\qquad
\sum_{k=1}^{K_t}\alpha_{t,k}^{*}=1.
\]

于是得到 item share target
\[
\alpha_t^* = (\alpha_{t,1}^*,\dots,\alpha_{t,K_t}^*).
\]

### 5.5 item advantage target

把 page advantage 分账到各 item:
\[
Y_{t,k}^{\mathrm{item}}
=
Y_t^{\mathrm{page}}\alpha_{t,k}^{*}.
\]

于是自动满足守恒:
\[
\sum_{k=1}^{K_t}Y_{t,k}^{\mathrm{item}}=Y_t^{\mathrm{page}}.
\]

这就是“item = page advantage 的可微分解”。

---

## 6. Item Allocator: 用 attention 学分账规律

为了在推理时不必每次都重新跑积分梯度，我们训练一个 `item allocator` 去拟合这种长期分账规律。

### 6.1 page context

构造 page context:
\[
c_t
=
\phi_{\mathrm{page}}
\Bigl(
[\,h_{t-1};\,h_t;\,h_t-h_{t-1};\,x_t^{\mathrm{page}}\,]
\Bigr),
\]
其中 \(x_t^{\mathrm{page}}\) 是页级统计特征。

### 6.2 item hidden

对每个 item 表示再做一次映射:
\[
u_{t,k}=\phi_{\mathrm{item}}(z_{t,k}).
\]

### 6.3 attention score

定义 item attention logit:
\[
e_{t,k}^{\mathrm{item}}
=
\frac{
(W_q c_t)^\top (W_k u_{t,k})
}{
\sqrt{d}
}
+
w_g^\top \sigma\!\left(
W_g[c_t;u_{t,k}]
\right).
\]

第一项是 query-key 匹配，刻画“当前页上下文和该 item 是否相配”。

第二项是门控项，允许模型学习更灵活的非线性交互。

### 6.4 item share 与 item advantage

对同页所有 item 做 softmax:
\[
\alpha_{t,k}
=
\mathrm{softmax}_k\!\left(e_{t,k}^{\mathrm{item}}\right).
\]

再由 page advantage 得到 item advantage 的预测:
\[
\hat A_{t,k}^{\mathrm{item}}
=
\alpha_{t,k}\hat A_t^{\mathrm{page}}.
\]

因此 `item allocator` 的职责不是重新预测一个完全独立的值，而是学会

- page credit 应该怎样在 item 之间分配
- 哪些 item 应该拿更多份额

### 6.5 item 层损失

\[
L_{\mathrm{item\text{-}share}}
=
\sum_t
\mathrm{KL}
\left(
\alpha_t^* \,\|\, \alpha_t
\right),
\]
\[
L_{\mathrm{item\text{-}adv}}
=
\sum_{t,k}
\ell\!\left(
\hat A_{t,k}^{\mathrm{item}},
Y_{t,k}^{\mathrm{item}}
\right).
\]

---

## 7. 为什么 token 要对 contextual hidden 做归因

如果 token 层直接对离散 token id 或静态 token embedding 做归因，会有三个问题:

1. 同一个 token 在不同 prefix 下意义不同。
2. 同一个 token 在不同 item 中作用不同。
3. 同一个 token 在不同用户状态下对长期价值的贡献也不同。

因此 token 层真正应该归因的对象不是裸 token，而是 decoder 在第 \(j\) 个位置形成的上下文化 hidden state:
\[
g_{t,k,j}
=
f_{\mathrm{tok}}
\left(
h_{t-1}, z_{t,k}, \tau_{t,k,<j}, \tau_{t,k,j}
\right).
\]

这里 \(g_{t,k,j}\) 表示:

- 当前用户状态是 \(h_{t-1}\)
- 当前 item 是 \(i_{t,k}\)
- 当前 prefix 是 \(\tau_{t,k,<j}\)
- 第 \(j\) 个 token 取值为 \(\tau_{t,k,j}\)

在这个具体上下文下，第 \(j\) 个 token 对长期价值的“实际语义作用”。

---

## 8. Token 粒度: 对 item advantage 做可微分账

### 8.1 teacher item critic

引入 teacher item critic:
\[
\mathcal{T}^{\mathrm{item}}_{\bar\theta}
\left(
h_{t-1},
z_{t,k},
g_{t,k,1:L_{t,k}}
\right)
\approx
Y_{t,k}^{\mathrm{item}}.
\]

它表示“当前状态下，这个 item 对长期价值的贡献是多少”。

### 8.2 token attribution score

记空白 token hidden 为 \(g_{\varnothing}\)。对第 \(j\) 个 token 的 attribution score 定义为:
\[
s_{t,k,j}^{\mathrm{tok}}
=
(g_{t,k,j}-g_{\varnothing})^\top
\int_0^1
\nabla_{g_{t,k,j}}
\mathcal{T}^{\mathrm{item}}_{\bar\theta}
\Bigl(
h_{t-1},
z_{t,k},
g_{t,k,1},
\dots,
g_{\varnothing}+\alpha(g_{t,k,j}-g_{\varnothing}),
\dots,
g_{t,k,L_{t,k}}
\Bigr)
d\alpha.
\]

它表达的是:

- 把第 \(j\) 个 token 从“空白 token”逐步插入到当前 prefix 中
- 观察 teacher item critic 对 item 长期价值判断如何变化
- 累积这个变化，得到该 token 的长期边际贡献

高效近似为
\[
s_{t,k,j}^{\mathrm{tok}}
\approx
\left\langle
\nabla_{g_{t,k,j}}
\mathcal{T}^{\mathrm{item}}_{\bar\theta},
\;
g_{t,k,j}-g_{\varnothing}
\right\rangle.
\]

### 8.3 sign-aware token share target

若 item 贡献为正:
\[
Y_{t,k}^{\mathrm{item}} \ge 0,
\]
则
\[
\tilde s_{t,k,j}^{\mathrm{tok}}
=
[s_{t,k,j}^{\mathrm{tok}}]_+.
\]

若 item 贡献为负:
\[
Y_{t,k}^{\mathrm{item}} < 0,
\]
则
\[
\tilde s_{t,k,j}^{\mathrm{tok}}
=
[-s_{t,k,j}^{\mathrm{tok}}]_+.
\]

归一化后得到 token share target:
\[
\beta_{t,k,j}^{*}
=
\frac{
\tilde s_{t,k,j}^{\mathrm{tok}}+\varepsilon
}{
\sum_{m=1}^{L_{t,k}}
\left(
\tilde s_{t,k,m}^{\mathrm{tok}}+\varepsilon
\right)
},
\qquad
\sum_{j=1}^{L_{t,k}}\beta_{t,k,j}^{*}=1.
\]

### 8.4 token advantage target

把 item advantage 分账到各 token:
\[
Y_{t,k,j}^{\mathrm{tok}}
=
Y_{t,k}^{\mathrm{item}} \beta_{t,k,j}^{*}.
\]

于是自动满足:
\[
\sum_{j=1}^{L_{t,k}}
Y_{t,k,j}^{\mathrm{tok}}
=
Y_{t,k}^{\mathrm{item}}.
\]

---

## 9. Tokenizer: 用 attention 学 token 分账规律

### 9.1 item-aware token context

首先构造 token 层的 item context:
\[
u_{t,k}^{\mathrm{ctx}}
=
\phi_{\mathrm{tok\text{-}item}}
\left(
[c_t;u_{t,k}]
\right).
\]

### 9.2 token hidden projection

对 token hidden 再映射:
\[
v_{t,k,j}
=
\phi_{\mathrm{tok\text{-}hid}}
\left(
[g_{t,k,j};E(\tau_{t,k,j})]
\right),
\]
其中 \(E(\tau_{t,k,j})\) 是 token embedding。

### 9.3 token attention score

定义 token logit:
\[
e_{t,k,j}^{\mathrm{tok}}
=
\frac{
(W_q^{\mathrm{tok}} u_{t,k}^{\mathrm{ctx}})^\top
(W_k^{\mathrm{tok}} v_{t,k,j})
}{
\sqrt{d}
}
+
{w_g^{\mathrm{tok}}}^\top
\sigma\!\left(
W_g^{\mathrm{tok}}
[u_{t,k}^{\mathrm{ctx}};v_{t,k,j}]
\right).
\]

### 9.4 token share 与 token advantage

对一个 item 内部所有 token 做 softmax:
\[
\beta_{t,k,j}
=
\mathrm{softmax}_j\!\left(
e_{t,k,j}^{\mathrm{tok}}
\right).
\]

于是 token advantage 预测为:
\[
\hat A_{t,k,j}^{\mathrm{tok}}
=
\beta_{t,k,j}\hat A_{t,k}^{\mathrm{item}}.
\]

### 9.5 token 层损失

\[
L_{\mathrm{tok\text{-}share}}
=
\sum_{t,k}
\mathrm{KL}
\left(
\beta_{t,k}^{*}\,\|\,\beta_{t,k}
\right),
\]
\[
L_{\mathrm{tok\text{-}adv}}
=
\sum_{t,k,j}
\ell\!\left(
\hat A_{t,k,j}^{\mathrm{tok}},
Y_{t,k,j}^{\mathrm{tok}}
\right).
\]

---

## 10. 三层守恒约束

这一套方案的核心不是“单独预测三个值”，而是让三层 credit 严格对应同一条长期价值链。

### 10.1 page 守恒

若 page bias 做了 session 内中心化，则
\[
\sum_{t=1}^{T} A_t^{\mathrm{page}}
=
Q^{\mathrm{sess}}(h_T)-Q^{\mathrm{sess}}(h_0).
\]

训练时可用预测量写成:
\[
L_{\mathrm{cons}}^{\mathrm{page}}
=
\left(
\sum_{t=1}^{T}\hat A_t^{\mathrm{page}}
-
\bigl(
\hat Q^{\mathrm{sess}}(h_T)-\hat Q^{\mathrm{sess}}(h_0)
\bigr)
\right)^2.
\]

### 10.2 item 守恒

\[
L_{\mathrm{cons}}^{\mathrm{item}}
=
\sum_t
\left(
\hat A_t^{\mathrm{page}}
-
\sum_{k=1}^{K_t}\hat A_{t,k}^{\mathrm{item}}
\right)^2.
\]

### 10.3 token 守恒

\[
L_{\mathrm{cons}}^{\mathrm{tok}}
=
\sum_{t,k}
\left(
\hat A_{t,k}^{\mathrm{item}}
-
\sum_{j=1}^{L_{t,k}}
\hat A_{t,k,j}^{\mathrm{tok}}
\right)^2.
\]

这三个守恒约束共同保证:

- page 不是“孤立预测”
- item 真的是对 page 的分账
- token 真的是对 item 的分账

---

## 11. 总损失

完整目标可以写为
\[
\begin{aligned}
L
=\;&
\lambda_{\mathrm{pre}} L_{\mathrm{pre}}
+
\lambda_{\mathrm{post}} L_{\mathrm{post}}
+
\lambda_{\mathrm{bias}} L_{\mathrm{bias}}
+
\lambda_{\mathrm{page}} L_{\mathrm{page}}
\\
&+
\lambda_{\mathrm{is}} L_{\mathrm{item\text{-}share}}
+
\lambda_{\mathrm{ia}} L_{\mathrm{item\text{-}adv}}
+
\lambda_{\mathrm{ts}} L_{\mathrm{tok\text{-}share}}
+
\lambda_{\mathrm{ta}} L_{\mathrm{tok\text{-}adv}}
\\
&+
\lambda_1 L_{\mathrm{cons}}^{\mathrm{page}}
+
\lambda_2 L_{\mathrm{cons}}^{\mathrm{item}}
+
\lambda_3 L_{\mathrm{cons}}^{\mathrm{tok}}.
\end{aligned}
\]

如果需要，也可以额外加入 teacher critic 的稳定项，但核心主目标就是上式。

---

## 12. Teacher 更新

teacher 的一个自然实现是 EMA:
\[
\bar\theta
\leftarrow
\rho \bar\theta + (1-\rho)\theta,
\qquad
\rho \in [0,1).
\]

其中:

- \(\theta\) 是当前 student critic 参数
- \(\bar\theta\) 是 teacher 参数
- \(\rho\) 越大，teacher 越平滑

这样做的目的有两个:

1. item/token pseudo-target 不会随 student 瞬时震荡而剧烈抖动。
2. 下层分账 target 始终来自一个更稳定的长期价值老师。

如果不想用 EMA，也可以每若干 step 冻结一次 teacher:
\[
\bar\theta \leftarrow \theta \quad \text{every } M \text{ steps}.
\]

---

## 13. 训练流程

一次 joint training 的顺序可以写成:

### Step 1. 构造 page 硬标签

对每个 session 计算
\[
G_t,\quad G_{t+1},\quad b_t,\quad Y_t^{\mathrm{page}}.
\]

### Step 2. 训练 session/page 头

更新
\[
\hat Q^{\mathrm{sess}}(h_{t-1}),
\quad
\hat Q^{\mathrm{sess}}(h_t),
\quad
\hat b_t,
\quad
\hat A_t^{\mathrm{page}}.
\]

### Step 3. 用 teacher page critic 生成 item pseudo-target

计算
\[
s_{t,k}^{\mathrm{item}}
\rightarrow
\alpha_{t,k}^{*}
\rightarrow
Y_{t,k}^{\mathrm{item}}.
\]

### Step 4. 训练 item allocator

更新
\[
\alpha_{t,k},
\qquad
\hat A_{t,k}^{\mathrm{item}}.
\]

### Step 5. 用 teacher item critic 生成 token pseudo-target

计算
\[
s_{t,k,j}^{\mathrm{tok}}
\rightarrow
\beta_{t,k,j}^{*}
\rightarrow
Y_{t,k,j}^{\mathrm{tok}}.
\]

### Step 6. 训练 token tokenizer

更新
\[
\beta_{t,k,j},
\qquad
\hat A_{t,k,j}^{\mathrm{tok}}.
\]

### Step 7. 施加三层守恒约束

联合优化
\[
L_{\mathrm{cons}}^{\mathrm{page}},
\quad
L_{\mathrm{cons}}^{\mathrm{item}},
\quad
L_{\mathrm{cons}}^{\mathrm{tok}}.
\]

### Step 8. 更新 teacher

执行 EMA:
\[
\bar\theta \leftarrow \rho \bar\theta + (1-\rho)\theta.
\]

---

## 14. 实践中的近似实现

完整公式里用到了积分梯度，但实际训练可以用下列近似:

### 14.1 Riemann 近似

对 item attribution:
\[
s_{t,k}^{\mathrm{item}}
\approx
\frac{1}{M}
\sum_{m=1}^{M}
(z_{t,k}-z_{\varnothing})^\top
\nabla_{z_{t,k}}
\mathcal{T}^{\mathrm{page}}_{\bar\theta}
\left(
z_{\varnothing}
\frac{m}{M}(z_{t,k}-z_{\varnothing})
\right).
\]

对 token attribution:
\[
s_{t,k,j}^{\mathrm{tok}}
\approx
\frac{1}{M}
\sum_{m=1}^{M}
(g_{t,k,j}-g_{\varnothing})^\top
\nabla_{g_{t,k,j}}
\mathcal{T}^{\mathrm{item}}_{\bar\theta}
\left(
g_{\varnothing}
\frac{m}{M}(g_{t,k,j}-g_{\varnothing})
\right).
\]

### 14.2 一阶近似

如果更强调吞吐量，可直接用
\[
\text{gradient} \times \text{input}
\]
近似:
\[
s_{t,k}^{\mathrm{item}}
\approx
\left\langle
\nabla_{z_{t,k}} \mathcal{T}^{\mathrm{page}}_{\bar\theta},
z_{t,k}-z_{\varnothing}
\right\rangle,
\]
\[
s_{t,k,j}^{\mathrm{tok}}
\approx
\left\langle
\nabla_{g_{t,k,j}} \mathcal{T}^{\mathrm{item}}_{\bar\theta},
g_{t,k,j}-g_{\varnothing}
\right\rangle.
\]

### 14.3 masked softmax

由于不同页的 \(K_t\) 不同，不同 item 的 \(L_{t,k}\) 也不同，share 归一化时需要 masked softmax:
\[
\alpha_{t,k}
=
\frac{
\exp(e_{t,k}^{\mathrm{item}})\mathbf{1}[k\le K_t]
}{
\sum_m \exp(e_{t,m}^{\mathrm{item}})\mathbf{1}[m\le K_t]
},
\]
\[
\beta_{t,k,j}
=
\frac{
\exp(e_{t,k,j}^{\mathrm{tok}})\mathbf{1}[j\le L_{t,k}]
}{
\sum_m \exp(e_{t,k,m}^{\mathrm{tok}})\mathbf{1}[m\le L_{t,k}]
}.
\]

---

## 15. 一个具体例子

假设第 \(t\) 页的 page hard target 为
\[
Y_t^{\mathrm{page}} = 0.12.
\]

这一页有 3 个 item，teacher page critic 给出的 item attribution score 为
\[
s_{t,1}^{\mathrm{item}}=2.0,\quad
s_{t,2}^{\mathrm{item}}=1.0,\quad
s_{t,3}^{\mathrm{item}}=0.5.
\]

因为 \(Y_t^{\mathrm{page}}>0\)，所以取正部并归一化:
\[
\alpha_t^* =
\left(
\frac{2.0}{3.5},
\frac{1.0}{3.5},
\frac{0.5}{3.5}
\right)
\approx
(0.571, 0.286, 0.143).
\]

于是 item target 为
\[
Y_{t,1}^{\mathrm{item}} \approx 0.0685,
\quad
Y_{t,2}^{\mathrm{item}} \approx 0.0343,
\quad
Y_{t,3}^{\mathrm{item}} \approx 0.0172.
\]

接着看第 1 个 item。假设它有 3 个 token，teacher item critic 给出的 token attribution score 为
\[
s_{t,1,1}^{\mathrm{tok}}=1.2,\quad
s_{t,1,2}^{\mathrm{tok}}=0.6,\quad
s_{t,1,3}^{\mathrm{tok}}=0.2.
\]

归一化后
\[
\beta_{t,1}^*
=
\left(
\frac{1.2}{2.0},
\frac{0.6}{2.0},
\frac{0.2}{2.0}
\right)
=
(0.6,0.3,0.1).
\]

于是 token target 为
\[
Y_{t,1,1}^{\mathrm{tok}} \approx 0.0411,
\quad
Y_{t,1,2}^{\mathrm{tok}} \approx 0.0206,
\quad
Y_{t,1,3}^{\mathrm{tok}} \approx 0.0069.
\]

可以看到:

- page credit 先分到账户级 item
- item credit 再分到账户级 token
- 每一层都严格守恒

---

## 16. 这一版方案的学术创新点

### 16.1 page 层是真长期价值，不是短期 pairwise margin

page 层直接建模
\[
Q^{\mathrm{sess}}(h_t)
\]
及其跨页差分，而不是仅依赖 pairwise preference。

### 16.2 item/token target 由长期价值模型自己反推

item 和 token 不再用手工 reward 规则硬拼，而是通过 teacher critic 的可微归因自动生成。

### 16.3 attention 不是普通打分，而是 amortized credit allocation

allocator/tokenizer 的角色不是简单分类器，而是学习“长期价值该怎么分账”的结构规律。

### 16.4 三层守恒把解释与价值建模绑在一起

守恒约束使得 page、item、token 不是三套彼此独立的预测头，而是一条统一的长期价值归因链。

---

## 17. 一句话总结

这套 TIGER-HCAA 的完整思想是:

1. 用 session 势函数差分定义 page 级长期贡献。
2. 用 teacher page critic 对 item 表示做可微归因，把 page credit 自动分到账户级 item。
3. 用 teacher item critic 对 token contextual hidden 做可微归因，把 item credit 自动分到账户级 token。
4. 用 attention allocator/tokenizer 学会这种分账规律。
5. 用三层守恒约束，把整个系统收敛为一条统一的长期价值归因链。

如果后续要继续扩成论文版，这一份公式已经可以直接作为“Method”部分的主干。
