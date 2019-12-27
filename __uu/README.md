# H1 markdown
## H2
### H3
#### H4
##### H5
###### H6

Alternatively, for H1 and H2, an underline-ish style:
[link](plt)
Alt-H1
======

Alt-H2
------

Emphasis, aka italics, with *asterisks* or _underscores_.

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~



1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

* Unordered list can use asterisks
- Or minuses
+ Or pluses
   
# add link   
http://github.com - automatic!
[GitHub](http://github.com)

# add pic
```text
![hello](picture file`abc.png`)
```
![hello](abc.png)


# As Kanye West said:

> We're living the future so
> the present is our past.


I think you should use an
`<addr>` element here instead.

```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```

```python
def foo():
    if not bar:
        return True
```
- [x] @mentions, #refs, [links](), **formatting**, and <del>tags</del> supported
- [x] list syntax required (any unordered or ordered list supported)
- [x] this is a complete item
- [ ] this is an incomplete item


First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column


# No name list
- uu.py
- abc.py
- util.py
- pytorch.py: no obvious .py names
- /pytorch: no obvious directory names

# gitignore everything but the folder
- In the target folder, create a .gitignore file
    - touch .gitignore
- put this in it.
    ```text
    *
    */
    !.gitignore
    ```
# git save username and password
- commands
    ```text
    git config --global credential.helper store
    git pull
    git push
    ```
    - username and passwords are saved in ```~/.git-credentials``` file.
    
    to view: ```vi filename```. ```ESC``` and ```:q``` or ```:wq``` to quit program.
- to undo this
    ```text
    git config --global --unset credential.helper
    ```
    then remove ```~/.git-credentials``` file.
    
    ```text
    rm ~/.git-credentials
    ```

# Delete all files in a folder EXCEPT 'pdf's.

```text
cd <the directory you want>
find . -type f ! -iname "*.pdf" -delete
```
- The first command will take you to the directory in which you want to delete your files
- The second command will delete all files except with those ending with .pdf in filename
- ```.``` is the current directory. ```!``` means to take all files except the ones with ```.pdf``` at the end. ```-type f``` selects only files, not directories. ```-delete``` means to delete it.

 **NOTE: this command will delete all files (except ```pdf``` files but including hidden files) in current directory as well as in all sub-directories. ```!``` must come before -name. simply ```-name``` will include only ```.pdf```, while ```-iname``` will include both ```.pdf``` and ```.PDF```**

To delete only in current directory and not in sub-directories add ```-maxdepth 1```:
```text
find . -maxdepth 1 -type f ! -iname "*.pdf" -delete
```
- I just realized, you can't use this to exclude 2 extensions, like ```*.py``` and ```*.md```.
I am moving to python codes.

# `hihihi`
- simple way of dark background color is `hihihi`.
- ``two works as well``.
- sure ```three also works```.