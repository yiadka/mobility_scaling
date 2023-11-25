# Scipyの使用方法
## Global optimization
潜在的に多くの局所的極小値が存在する中で、与えられた境界内で関数の大域的極小値を求める。


## curve_fit
使用するパラメータ
- f  使用するモデル
- xdata
独立変数。`float`変換可能でないといけない
- ydata
従属変数。`f(xdata, ...)`
- p0
パラメータの初期値
- sigma 
指定する際には`chisq = sum((r / sigma)**2)`という関数を使って指定する
- method 
制約の問題は`lm`、境界が与えられているならば、`trf`が良い

