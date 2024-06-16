-- CreateTable
CREATE TABLE "Assistants" (
    "id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "parameters" JSONB NOT NULL,

    CONSTRAINT "Assistants_pkey" PRIMARY KEY ("id")
);
