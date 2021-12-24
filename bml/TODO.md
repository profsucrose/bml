# TODO

* Matrices
* Fix nested blocks in macros
* Sc integration w/ bot
* Gif generation w/ bot
* add flags for blank image generation


## BML
    * DONE: finish preprocessor (nested macros)
    * DONE: test AST, hook up w/ image library
    * DONE: Integrate lasso
    * DONE: PARSER
        - frag, coord, resolution, frame, max_frame
    * Extra lang features:
        - more built-in functions (GLSL)
        - matrices
        - repeat block
    * Maybe: add pretty error reporting w/ codespan
        - parser
        - AST eval?
    * TCP daemon/server for processing PNGs, JPGs & gifs?
        - integrate bml w/ bot

## WORKING: SlackBot
    * take money for people to manipulate image
        - needs own token
    * /bot corrupt "script.bml"
        - gets last posted image
        - runs bml
        - posts in channel 
    * *bot listens for DMs*
        - if receives script.bml, uploads script
    * every min you get 5
    * gifs


