function(clue_finalize tgt)
    # direct dependency as workaround
    target_link_libraries(${tgt} PRIVATE alpaka::alpaka)
    alpaka_finalize(${tgt})
endfunction()