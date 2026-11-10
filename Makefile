.SUFFIXES: .go

TARGETS=multauct version.go
SOURCES=multauct.go utility.go

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

version.go: $(SOURCES)
	./mkversion $(SOURCES)

multauct: $(SOURCES) version.go
	go build -o multauct $(SOURCES) version.go

install: multauct
	cp multauct $(HOME)/bin

.go:
	go build $<
