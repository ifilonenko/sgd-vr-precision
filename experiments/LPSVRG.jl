"""

This struct is responsible for leveraging the LP-SVRG algorithm. The input for
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
   g_l :: (generic function) :: gradient loss function you wish to minimize
      This function must take in (wtilde,i)
         - wtilde, new w value per epoch
         - i,index

# Examples
```julia-repl
julia> B = Scaled{Int8,7,1.0/128.0,Randomized}
julia> LPSVRG{alpha,niters,nepochs,B}(w0,wopt,X,Y,g_l)
...
```
"""
struct LPSVRG{a,T,K,B <: Scaled} <: Quantized{a,T,K,B}
   LPSVRG{a,T,K,B}(w0::AbstractArray{F,N},wopt::AbstractArray{F,N},
   X::AbstractArray{F,N2},Y::AbstractArray{F2,N},g_l) where
      {F<:Number,F2<:Number,B <: Scaled,a,T,K,N,N2} =
      run_algo(LPSVRG{a,T,K,B},w0,wopt,X,Y,g_l)
end
function run_algo(::Type{LPSVRG{a,T,K,B}},w0,wopt,X,Y,g_l) where
   {B <: Scaled,a,T,K}
      w = B(w0)
      (N, d) = size(X)
      dist_to_optimum = zeros(T*K)
      for k = 1:K
         wtilde = B(w);
         gtilde = mapreduce(i->g_l(wtilde)*X[i,:], +, 1:N)
         w = wtilde;
         for t = 1:T
            i = rand(1:N)
            w = B(w - a*(g_l(w,i,X,Y) - g_l(wtilde,i,X,Y) + gtilde))
            dist_to_optimum[k + (epi-1)*niters] = norm(w - wopt);
         end
      end
      return (w, dist_to_optimum)
end
