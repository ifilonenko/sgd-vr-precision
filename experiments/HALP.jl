"""

This struct is responsible for leveraging the HALP algorithm. The input for
the algorithm are the following:
Inputs:
   a :: Real, the alpha value which correlates to step size
   T :: Integer, number of iterations of applying random sampling of gradient
   K :: Integer, number of epochs of applying a full gradient
   B :: Scaled, a FixedPointNumbers.Scaled type use to quantize (blue box)

   w0 :: Array{Float64}, the initial randomized iterate
   wopt :: Array{Float64}, optimal w value take for distance purposes
   X :: Array{Float64}, training X values (input features)
   Y :: Array{Float64}, training Y values (labels)
   mu :: Float64, Scale factor
   g_l :: (generic function) :: gradient loss function you wish to minimize
      This function must take in (wtilde,X,Y)
         - wtilde, new w value per epoch
         - X, training data
         - Y, training labels

# Examples
```julia-repl
julia> B = Scaled{Int8,7,1.0/128.0,Randomized}
julia> LPSVRG{alpha,niters,nepochs,B}(w0,wopt,X,Y,g_l)
...
```
"""
struct HALP{a,T,K,B <: Scaled} <: Quantized{a,T,K,B}
   HALP{a,T,K,B}(w0::AbstractArray{F,N},wopt::AbstractArray{F,N},
   X::AbstractArray{F,N2},Y::AbstractArray{F2,N},mu::Real,g_l) where
      {F<:Number,F2<:Number,B <: Scaled,a,T,K,N,N2} =
      run_algo(LPSVRG{a,T,K,B},w0,wopt,X,Y,mu,g_l)
end
function run_algo(::Type{LPSVRG{a,T,K,B}},w0,wopt,X,Y,mu,g_l) where {a,T,K,B}
   wtilde = w0
   (N, d) = size(X)
   z = zeros(d)
   dist_to_optimum = zeros(T*K)
   blue_box = B
   for k = 1:K
      wtilde = wtilde + z
      gtilde = mapreduce(i->g_l(wtilde,X[i,:]',Y[i]), +, 1:N) / N
      s = norm(gtilde)/(mu*(2^(get_f(blue_box))-1))
      blue_box = Scaled{get_T(B),get_f(B),s,get_r(B)}
      z = zeros(d)
      for t = 1:T
         i = rand(1:N)
         xi = X[i,:]';
         yi = Y[i];
         z = blue_box(z - a*(g_l(wtilde+z,xi,yi) - g_l(wtilde,xi,yi) + gtilde))
         dist_to_optimum[t+((k-1)*T)] = norm(wtilde+z-wopt)
      end
   end
   return (wtilde+z, dist_to_optimum)
end
