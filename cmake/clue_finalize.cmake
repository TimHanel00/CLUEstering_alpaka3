function(clue_finalize tgt)
    # make alpaka a direct dependency so alpaka_finalize can mark the target
    target_link_libraries(${tgt} PRIVATE alpaka::alpaka)
    alpaka_finalize(${tgt})
endfunction()