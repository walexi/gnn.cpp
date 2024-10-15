
OUT = bin
SRCDIR = src/
INCDIR = include/
OBJDIR = .obj/

CPPFILES = $(wildcard $(SRCDIR).cpp)

CPPOBJS = $(patsubst $(SRCDIR)%.cpp,$(OBJDIR)%.o,$(CPPFILES))

DEPS = $(CPPOBJS:.o=.d)

CPP=g++

CPPFLAGS=-g -Wall -I$(INCDIR)

all: $(OBJDIR) $(OUT)

$(OUT): $(CPPOBJS)
	$(CPP) $(CPPFLAGS) -o $(OUT) $(CPPOBJS)

-include $(DEPS)

$(CPPOBJS):
	$(CPP) $(CPPFLAGS) -c $(patsubst $(OBJDIR)%.o,$(SRCDIR)%.cpp,$(@)) -o $(@)


$(OBJDIR):
	mkdir -p $@

$(OUT):
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(CPPOBJS) $(DEPS) $(OUT) $(OBJDIR)