package main

// match & search

import "fmt"

func IsThis(verbal, visual chan rune) (chan rune) {
    out := make(chan rune)

    go func() {
        text := ""
        for c := range verbal {
            text += string(c)
            if text == "is this a banana?" {
                viz := ""
                for v := range visual {
                    viz += string(v)
                }

                res := ""
                if viz == "banana" {
                    res = "yes"
                } else {
                    res = "no"
                }

                for _, r := range res {
                    out <- r
                }
            }
        }
    }()

    return out
}

func main() {
    verbal := make(chan rune, 100)
    visual := make(chan rune, 100)

    for _, c := range "is this a banana?" {
        verbal <- c
    }
    close(verbal)

    for _, c := range "banana" {
        visual <- c
    }
    close(visual)

    out := IsThis(verbal, visual)
    for c := range out {
        fmt.Printf("%s", string(c))
    }

}
