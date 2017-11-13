# __precompile__()
module GradientModels
    using FixedPointNumbers
    abstract type NonQuantized{alpha,T,K} <: Real end
    abstract type Quantized{alpha,T,K,S <: Scaled} <: Real end
export NonQuantized,
    Quantized,
    SVRG,
    LPSVRG,
    LMHALP
include("SVRG.jl");
include("LPSVRG.jl");
include("LMHALP.jl");
end # module
