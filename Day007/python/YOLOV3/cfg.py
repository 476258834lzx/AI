IMG_HEIGHT=416
IMG_WIDTH=416

CLASS_NUM=5

ANCHORS={
    13:[[29,35], [49,36], [59,50]],
    26:[[18,12], [19,23], [36,19]],
    52:[[3, 6], [8, 9], [10, 16]]
}

ANCHORS_AREA={
    13:[x*y for x,y in ANCHORS[13]],
    26:[x*y for x,y in ANCHORS[26]],
    52:[x*y for x,y in ANCHORS[52]]
}
