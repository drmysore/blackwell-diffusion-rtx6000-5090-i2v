;;; .dir-locals.el
((c++-ts-mode
  . ((eval . (let ((ws-root (locate-dominating-file default-directory "MODULE.bazel")))
               ;; ensure the hook variable is bound before using it
               (require 'compile)
               (setq-local compilation-directory ws-root)
               (setq-local compile-command
                           (format "cd %s && bazel build //..." ws-root))

               (unless (assoc 'c++-ts-mode
                              (default-value 'compilation-filename-transformers))
                 (add-to-list 'compilation-filename-transformers
                              `(lambda (filename _)
                                 (unless (file-name-absolute-p filename)
                                   (expand-file-name filename ,ws-root))))))))))
